import figs.render.capture_generation as pg
import figs.visualize.generate_videos as gv

from figs.simulator import Simulator
from figs.control.vehicle_rate_mpc import VehicleRateMPC
import logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("figs")
logger.setLevel(logging.INFO)

# capture_name="aerialarena_20260401_165257_007_icosahedral_bigsubset"
# capture_name="aerialarena_square_subset"
capture_name="aerialarena_square_16x"
# capture_name = "aerial_arena_subset"
# capture_name = "aerialarena_20260401_165257_007_icosahedral_full"
# capture_name = "aerialarena_20260401_165257_007_icosahedral"
pg.generate_gsplat(
    scene_file_name=capture_name,
    capture_cfg_name="rig_hloc",
    use_images=True,
)