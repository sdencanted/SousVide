import inspect
import os
import tempfile
import unittest
from unittest import mock

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
    _move_to_device,
    _validate_dataloader_num_workers,
    train_roster,
    train_student,
)


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

    def test_worker_count_must_be_a_non_negative_integer(self):
        for invalid_value in (-1,1.5,"4",True):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError,"non-negative integer"):
                    _validate_dataloader_num_workers(invalid_value)

        for valid_value in (0,4):
            _validate_dataloader_num_workers(valid_value)

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
