import copy
import inspect
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import sousvide.instruct.train_policy as train_policy
from sousvide.instruct.synthesized_data import (
    OBSERVATION_FORMAT_VERSION,
    FileGroupedBatchSampler,
    generate_dataset,
    stack_samples,
)
from sousvide.instruct.train_policy import (
    _compile_network,
    _create_dataloader,
    _device_batches,
    _move_to_device,
    _new_phase_timings,
    _validate_dataloader_num_workers,
    _validate_numerical_mode_options,
    train_roster,
    train_student,
)
from sousvide.instruct.losses import LossFn


class _TinyHistNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(4,6),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(6,2),
        )

    def forward(self,inputs):
        dynamics = torch.flatten(inputs["dynamics"],start_dim=-2)
        return {"parameters":self.layers(dynamics)}


def _run_legacy_reference(network,train_paths,test_paths,epochs,batch_size):
    optimizer = torch.optim.Adam(network.parameters(),lr=1e-4)
    criterion = LossFn()
    train_losses,test_losses = [],[]
    train_count = test_count = 0
    for epoch in range(epochs):
        epoch_train_losses,train_count = [],0
        for path in train_paths:
            loader = torch.utils.data.DataLoader(
                generate_dataset(path),batch_size=batch_size,
                shuffle=True,drop_last=False)
            for inputs,labels in loader:
                predictions = network(inputs)
                loss = criterion(predictions,labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_train_losses.append(batch_size*loss.item())
                train_count += batch_size

        epoch_test_losses,test_count = [],0
        for path in test_paths:
            loader = torch.utils.data.DataLoader(
                generate_dataset(path),batch_size=batch_size,
                shuffle=True,drop_last=False)
            for inputs,labels in loader:
                predictions = network(inputs)
                loss = criterion(predictions,labels)
                epoch_test_losses.append(batch_size*loss.item())
                test_count += batch_size

        train_losses.append(
            (epoch+1,sum(epoch_train_losses)/train_count))
        test_losses.append(
            (epoch+1,sum(epoch_test_losses)/test_count))

    return {
        "Loss_tn":np.array(train_losses).T,
        "Loss_tt":np.array(test_losses).T,
        "Nd_tn":train_count,
        "Nd_tt":test_count,
    }


class TrainingDataLoaderTests(unittest.TestCase):
    class _Progress:
        def __enter__(self):
            return self

        def __exit__(self,*args):
            return False

        def add_task(self,*args,**kwargs):
            return 1

        def refresh(self):
            pass

    def test_training_entrypoints_default_to_four_workers(self):
        self.assertEqual(
            inspect.signature(train_roster).parameters[
                "dataloader_num_workers"
            ].default,
            4,
        )

    def test_train_roster_forwards_worker_override(self):
        progress = self._Progress()
        console = mock.Mock()
        with (
            mock.patch.object(train_policy.ru,"get_console",return_value=console),
            mock.patch.object(
                train_policy.ru,"get_training_progress",return_value=progress),
            mock.patch.object(train_policy,"train_student") as train_student_mock,
        ):
            train_roster(
                "cohort",["student"],"network",1,
                dataloader_num_workers=7)

        self.assertEqual(
            train_student_mock.call_args.kwargs["dataloader_num_workers"],7)
        self.assertEqual(
            train_student_mock.call_args.kwargs["numerical_mode"],"modern")
        self.assertEqual(
            inspect.signature(train_student).parameters[
                "dataloader_num_workers"
            ].default,
            4,
        )

    def test_missing_regeneration_only_generates_absent_student_data(self):
        progress = self._Progress()
        with (
            mock.patch.object(
                train_policy.ru,"get_console",return_value=mock.Mock()),
            mock.patch.object(
                train_policy.ru,"get_training_progress",return_value=progress),
            mock.patch.object(
                train_policy,"_observation_data_available",
                side_effect=[True,False]),
            mock.patch.object(
                train_policy.og,"generate_observation_data") as generate,
            mock.patch.object(train_policy,"train_student"),
        ):
            train_roster(
                "cohort",["ready","missing"],"commNet",1,
                regen="missing",image_modality="kronecker_delta")

        generate.assert_called_once_with(
            "cohort",["missing"],networks=["commNet"],
            image_modality="kronecker_delta")

    def test_dataloader_uses_configured_worker_and_pinning_options(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(4))

        multi_worker_loader = _create_dataloader(dataset,2,True,4,True)
        single_worker_loader = _create_dataloader(dataset,2,False,0,False)

        self.assertEqual(multi_worker_loader.num_workers,4)
        self.assertTrue(multi_worker_loader.pin_memory)
        self.assertIsNotNone(multi_worker_loader.sampler)
        self.assertEqual(single_worker_loader.num_workers,0)
        self.assertFalse(single_worker_loader.pin_memory)

    def test_persistent_workers_are_only_enabled_for_worker_processes(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(4))
        persistent = _create_dataloader(
            dataset,2,False,2,False,persistent_workers=True)
        synchronous = _create_dataloader(
            dataset,2,False,0,False,persistent_workers=True)

        self.assertTrue(persistent.persistent_workers)
        self.assertFalse(synchronous.persistent_workers)

    def test_multiple_workers_prepare_cpu_batches(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(8))
        dataloader = _create_dataloader(dataset,2,False,2,False)

        batches = [batch[0] for batch in dataloader]

        self.assertEqual(torch.cat(batches).tolist(),list(range(8)))
        self.assertTrue(all(batch.device.type == "cpu" for batch in batches))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA is unavailable")
    def test_cuda_prefetch_preserves_batch_values(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(12).reshape(6,2))
        dataloader = _create_dataloader(dataset,2,False,2,True)
        timings = _new_phase_timings()

        batches = list(_device_batches(
            dataloader,torch.device("cuda:0"),True,True,timings))

        values = torch.cat([batch[0].cpu() for batch in batches])
        torch.testing.assert_close(values,torch.arange(12).reshape(6,2))
        self.assertTrue(all(batch[0].is_cuda for batch in batches))

    def test_worker_count_must_be_a_non_negative_integer(self):
        for invalid_value in (-1,1.5,"4",True):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError,"non-negative integer"):
                    _validate_dataloader_num_workers(invalid_value)

        for valid_value in (0,4):
            _validate_dataloader_num_workers(valid_value)

    def test_original_numerical_mode_rejects_incompatible_options(self):
        incompatible = [
            {"network_name":"commNet"},
            {"regen":True},
            {"deployment":("course","scene","method")},
            {"image_modality":"kronecker_delta"},
            {"compile_mode":"default"},
            {"precision":"bfloat16"},
            {"persistent_dataloader":True},
            {"seed":2},
        ]
        defaults = {
            "network_name":"histNet",
            "numerical_mode":"original",
            "regen":False,
            "deployment":None,
            "image_modality":"rgb",
            "compile_mode":"none",
            "precision":"float32",
            "persistent_dataloader":False,
            "seed":0,
        }
        for override in incompatible:
            with self.subTest(override=override):
                options = defaults|override
                with self.assertRaisesRegex(
                        ValueError,"original numerical mode requires"):
                    _validate_numerical_mode_options(**options)

        _validate_numerical_mode_options(**defaults)

    def test_original_numerical_mode_matches_legacy_weights_rng_and_logs(self):
        batch_size,epochs = 3,2
        samples = [
            {"dynamics":torch.arange(
                index*4,(index+1)*4,dtype=torch.float32).reshape(1,2,2)/10}
            for index in range(14)
        ]
        labels = [
            {"parameters":torch.tensor(
                [[index/7,(index+1)/9]],dtype=torch.float32)}
            for index in range(14)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_path = os.path.join(temp_dir,"legacy.pt")
            packed_path = os.path.join(temp_dir,"packed.pt")
            test_path = os.path.join(temp_dir,"test.pt")
            torch.save(
                {"Xnn":samples[:5],"Ynn":labels[:5]},legacy_path)
            torch.save(
                {
                    "format_version":OBSERVATION_FORMAT_VERSION,
                    "Xnn":stack_samples(samples[5:9]),
                    "Ynn":stack_samples(labels[5:9]),
                },
                packed_path,
            )
            torch.save(
                {
                    "format_version":OBSERVATION_FORMAT_VERSION,
                    "Xnn":stack_samples(samples[9:]),
                    "Ynn":stack_samples(labels[9:]),
                },
                test_path,
            )
            train_paths = [legacy_path,packed_path]
            test_paths = [test_path]

            torch.manual_seed(81)
            base_network = _TinyHistNet()
            initial_rng = torch.get_rng_state().clone()

            for worker_count in (0,2):
                with self.subTest(worker_count=worker_count):
                    reference_network = copy.deepcopy(base_network)
                    compatible_network = copy.deepcopy(base_network)
                    torch.set_rng_state(initial_rng)
                    reference = _run_legacy_reference(
                        reference_network,train_paths,test_paths,
                        epochs,batch_size)
                    reference_rng = torch.get_rng_state().clone()

                    run_path = os.path.join(
                        temp_dir,f"workers-{worker_count}")
                    os.makedirs(run_path)
                    torch.set_rng_state(initial_rng)
                    network_path = os.path.join(run_path,"histNet.pt")
                    student = SimpleNamespace(
                        name="student",
                        path=run_path,
                        policy=SimpleNamespace(
                            networks={"histNet":compatible_network},
                            network_paths={"histNet":network_path},
                        ),
                    )
                    with (
                        mock.patch.object(
                            train_policy,"Pilot",return_value=student),
                        mock.patch.object(
                            train_policy,"get_data_paths",
                            return_value=(train_paths,test_paths)),
                        mock.patch.object(
                            train_policy.torch.cuda,"is_available",
                            return_value=False),
                    ):
                        train_student(
                            "cohort","student","histNet",epochs,
                            lim_sv=epochs,batch_size=batch_size,
                            dataloader_num_workers=worker_count,
                            numerical_mode="original")

                    compatible_rng = torch.get_rng_state().clone()
                    for expected,actual in zip(
                            reference_network.parameters(),
                            compatible_network.parameters()):
                        self.assertTrue(torch.equal(expected,actual))
                    self.assertTrue(torch.equal(
                        reference_rng,compatible_rng))

                    losses = torch.load(
                        os.path.join(run_path,"losses_histNet.pt"),
                        weights_only=False)
                    entry = next(iter(losses.values()))
                    np.testing.assert_array_equal(
                        reference["Loss_tn"],entry["Loss_tn"])
                    np.testing.assert_array_equal(
                        reference["Loss_tt"],entry["Loss_tt"])
                    self.assertEqual(reference["Nd_tn"],entry["Nd_tn"])
                    self.assertEqual(reference["Nd_tt"],entry["Nd_tt"])
                    self.assertEqual(entry["numerical_mode"],"original")
                    self.assertEqual(entry["Eval_tte"].shape,(0,2))

    def test_nested_batch_moves_without_changing_container_structure(self):
        batch = {
            "inputs": [torch.ones(2),{"history": torch.zeros(1)}],
            "labels": (torch.tensor([2.0]),"metadata"),
        }

        moved = _move_to_device(batch,torch.device("cpu"))

        self.assertIsInstance(moved,dict)
        self.assertIsInstance(moved["inputs"],list)
        self.assertIsInstance(moved["inputs"][1],dict)
        self.assertIsInstance(moved["labels"],tuple)
        self.assertEqual(moved["labels"][1],"metadata")
        self.assertEqual(moved["inputs"][0].device.type,"cpu")

    def test_generated_dataset_is_cpu_resident(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir,"observations.pt")
            torch.save(
                {
                    "Xnn": [torch.ones(1,2)],
                    "Ynn": [torch.zeros(1,1)],
                },
                data_path,
            )

            dataset = generate_dataset(data_path)
            inputs,labels = dataset[0]

        self.assertEqual(inputs.device.type,"cpu")
        self.assertEqual(labels.device.type,"cpu")

    def test_contiguous_and_legacy_datasets_return_identical_samples(self):
        samples = [
            {"image":torch.full((1,2,2),float(index)),
             "state":torch.tensor([[index,index+1.0]])}
            for index in range(3)
        ]
        labels = [
            {"command":torch.tensor([[index*2.0]])}
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_path = os.path.join(temp_dir,"legacy.pt")
            packed_path = os.path.join(temp_dir,"packed.pt")
            torch.save({"Xnn":samples,"Ynn":labels},legacy_path)
            torch.save(
                {
                    "format_version":OBSERVATION_FORMAT_VERSION,
                    "Xnn":stack_samples(samples),
                    "Ynn":stack_samples(labels),
                },
                packed_path,
            )
            legacy = generate_dataset(legacy_path)
            packed = generate_dataset(packed_path)

            self.assertEqual(len(legacy),len(packed))
            for index in range(len(legacy)):
                legacy_input,legacy_label = legacy[index]
                packed_input,packed_label = packed[index]
                for key in legacy_input:
                    torch.testing.assert_close(
                        legacy_input[key],packed_input[key])
                torch.testing.assert_close(
                    legacy_label["command"],packed_label["command"])

    def test_file_grouped_sampler_preserves_file_local_batches(self):
        datasets = [
            torch.utils.data.TensorDataset(torch.arange(3)),
            torch.utils.data.TensorDataset(torch.arange(4)),
        ]
        sampler = FileGroupedBatchSampler(
            datasets,batch_size=2,shuffle=False)

        self.assertEqual(
            list(sampler),[[0,1],[2],[3,4],[5,6]])
        self.assertEqual(len(sampler),4)

    def test_compile_wrapper_does_not_replace_original_network(self):
        network = torch.nn.Linear(2,1)
        compiled = object()
        with mock.patch.object(
                train_policy.torch,"compile",return_value=compiled) as compile_mock:
            result = _compile_network(network,"reduce-overhead")

        self.assertIs(result,compiled)
        compile_mock.assert_called_once_with(network,mode="reduce-overhead")


if __name__ == "__main__":
    unittest.main()
