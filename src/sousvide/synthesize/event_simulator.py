"""FiGS simulator variant that exposes simulation-rate frames to v2e."""

from __future__ import annotations

import numpy as np

import figs.dynamics.quadcopter_specifications as qs
import figs.utilities.orientation_helper as oh
import figs.utilities.transform_helper as th

from figs.dynamics.external_forces import ExternalForces
from figs.simulator import Simulator
from tqdm.auto import tqdm
from sousvide.synthesize.image_modality import (
    ImageModality,kronecker_to_three_channels,validate_image_modality,
)


class EventSimulator(Simulator):
    """A Simulator with a hidden event warm-up and frame callback.

    The base FiGS simulator is intentionally left untouched. This subclass is
    selected for event-enabled rollout generation and online deployment.
    """

    def simulate_with_events(
        self,policy,t0,tf,x0,event_frame_callback,warmup_steps,
        warmup_policy=None,image_modality:ImageModality="rgb",
    ):
        nw = 6
        rollout = self.conFiG["rollout"]
        spec = qs.generate_specifications(self.conFiG["frame"])
        fex = ExternalForces(self.conFiG["forces"])

        nx, nu = spec["nx"], spec["nu"]
        m, kt = spec["m"], spec["kt"]
        g, nrtr = spec["g"], spec["Nrtr"]
        tc2b = spec["Tc2b"]
        rgb_dim, dpt_dim = spec["rgb_dim"], spec["dpt_dim"]
        camera = self.gsplat.generate_output_camera(spec["camera"])

        hz_sim = rollout["frequency"]
        model_noise = rollout["noise"]["model"]
        sensor_noise = rollout["noise"]["sensor"]

        if model_noise is None:
            mu_md_s, std_md_s = np.zeros(nx), np.zeros(nx)
        else:
            mu_md_s = np.array(model_noise["mean"])
            std_md_s = np.array(model_noise["std"])

        if sensor_noise is None:
            mu_sn, std_sn = np.zeros(nx), np.zeros(nx)
        else:
            mu_sn = np.array(sensor_noise["mean"])
            std_sn = np.array(sensor_noise["std"])

        image_modality = validate_image_modality(image_modality)
        warmup_policy = warmup_policy or policy
        n_sim2ctl = int(hz_sim / policy.hz)
        if n_sim2ctl * policy.hz != hz_sim:
            raise ValueError("Simulation frequency must be divisible by controller frequency.")
        if warmup_policy.hz != policy.hz:
            raise ValueError("Warm-up and evaluated policies must use the same frequency.")
        if warmup_steps != n_sim2ctl:
            raise ValueError(
                f"Event warm-up must cover one control interval ({n_sim2ctl} simulation steps)."
            )

        mu_md = mu_md_s * (1 / n_sim2ctl)
        std_md = std_md_s * (1 / n_sim2ctl)
        duration = np.round(tf - t0, 5)
        nsim = int(duration * hz_sim)
        nctl = int(duration * policy.hz)

        tro = np.zeros((nctl + 1))
        xro = np.zeros((nctl + 1, nx))
        uro = np.zeros((nctl, nu))
        wro = np.zeros((nctl, nw))
        rgb_ro = np.zeros(((nctl,) + rgb_dim), dtype=np.uint8)
        dpt_ro = np.zeros(((nctl,) + dpt_dim), dtype=np.uint8)
        tsol_ro = np.zeros((nctl,))

        xcr, xpr = x0.copy(), x0.copy()
        ucr = np.array([-(m * g) / (nrtr * kt), 0.0, 0.0, 0.0])
        tau_cr = np.zeros(3)

        # The sampled perturbation is one control interval before t0. These
        # frames/states are deliberately not written to rollout arrays.
        t_warmup = t0 - warmup_steps / hz_sim
        for i in range(warmup_steps):
            tcr = t_warmup + i / hz_sim
            fcr = fex.get_forces(xcr[0:6], noisy=True)
            pcr = np.hstack((m, kt, fcr))
            fts = np.hstack((fcr, tau_cr))
            is_control_step = i % n_sim2ctl == 0
            if event_frame_callback is not None or is_control_step:
                tb2w = th.x_to_T(xcr)
                rgb,dpt = self.gsplat.render_rgb(camera,tb2w @ tc2b)
                if event_frame_callback is not None:
                    event_frame_callback(rgb,i/hz_sim,False)

            if is_control_step:
                xsn = xcr + np.random.normal(loc=mu_sn, scale=std_sn)
                xsn[6:10] = oh.obedient_quaternion(xsn[6:10], xpr[6:10])
                ucr,_ = warmup_policy.control(tcr,xsn,ucr,rgb,dpt,fts)

            xpr = xcr
            xcr = self.solver.simulate(x=xcr, u=ucr, p=pcr)
            xcr = xcr + np.random.normal(loc=mu_md, scale=std_md)
            xcr[6:10] = oh.obedient_quaternion(xcr[6:10], xpr[6:10])

        # The last event frame is the last saved RGB frame. The remaining four
        # integrations only produce the terminal trajectory state.
        final_event_step = nsim - n_sim2ctl
        policy_name = getattr(policy,"name",policy.__class__.__name__)
        for i in tqdm(range(nsim), desc=f"Simulating {policy_name} rollout", total=nsim,leave=False):
            tcr = t0 + i / hz_sim
            fcr = fex.get_forces(xcr[0:6], noisy=True)
            pcr = np.hstack((m, kt, fcr))
            fts = np.hstack((fcr, tau_cr))

            if i % n_sim2ctl == 0:
                tb2w = th.x_to_T(xcr)
                rgb, dpt = self.gsplat.render_rgb(camera, tb2w @ tc2b)
                xsn = xcr + np.random.normal(loc=mu_sn, scale=std_sn)
                xsn[6:10] = oh.obedient_quaternion(xsn[6:10], xpr[6:10])

                if image_modality == "kronecker_delta":
                    if event_frame_callback is None:
                        raise ValueError(
                            "Kronecker deployment requires an event frame callback.")
                    event_image = event_frame_callback(
                        rgb,(warmup_steps+i)/hz_sim,True)
                    if event_image is None:
                        raise RuntimeError(
                            "Event callback did not return a control-boundary image.")
                    policy_image = kronecker_to_three_channels(event_image)
                else:
                    policy_image = rgb

                ucr,tsol = policy.control(
                    tcr,xsn,ucr,policy_image,dpt,fts)

                k = i // n_sim2ctl
                tro[k], xro[k, :], uro[k, :] = tcr, xcr, ucr
                wro[k, 0:3] = fcr
                rgb_ro[k, ...], dpt_ro[k, ...] = rgb, dpt
                tsol_ro[k] = sum(tsol.values())

                if (image_modality == "rgb" and
                    event_frame_callback is not None and i <= final_event_step):
                    event_frame_callback(rgb, (warmup_steps + i) / hz_sim, True)
            elif event_frame_callback is not None and i <= final_event_step:
                tb2w = th.x_to_T(xcr)
                rgb, _ = self.gsplat.render_rgb(camera, tb2w @ tc2b)
                event_frame_callback(rgb, (warmup_steps + i) / hz_sim, False)

            xpr = xcr
            xcr = self.solver.simulate(x=xcr, u=ucr, p=pcr)
            xcr = xcr + np.random.normal(loc=mu_md, scale=std_md)
            xcr[6:10] = oh.obedient_quaternion(xcr[6:10], xpr[6:10])

        tro[nctl] = t0 + nsim / hz_sim
        xro[nctl, :] = xcr
        return tro, xro, uro, wro, rgb_ro, dpt_ro, tsol_ro
