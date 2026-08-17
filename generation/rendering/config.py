# ========== SCENE SETTINGS ==========
CLEAR_SCENE = True  # clear scene before each render
UNITS = "METRIC"
SCALE = 1

# ========== RENDER SETTINGS ==========
RENDER_ENGINE = "CYCLES"  # 'CYCLES', 'BLENDER_EEVEE_NEXT'
RENDER_SAMPLES = 128
USE_DENOISING = True  # CYCLES only
RESOLUTION_X = 1024
RESOLUTION_Y = 1440
RESOLUTION_PERCENTAGE = 100  # scale factor applied to the resolution above
FILE_FORMAT = "PNG"
COLOR_DEPTH = "8"  # '8', '16'
COMPRESSION = 15  # 0-100
SMOOTH_SHADING = True
AUTO_SMOOTH = True
SHADING_ANGLE = 25

# ========== OBJECT SETTINGS ==========
PAPER_OBJECT_NAME = "Paper"  # name given to the imported mesh, used to look it up later
PAPER_LOCATION = (0, 0, 0)
PAPER_ROTATION = (0, 0, 0)
PAPER_SCALE = (1, 1, 1)

# Table/Surface settings
TABLE_OBJECT_NAME = "Table"  # name given to the surface plane, used to look it up later
TABLE_SIZE = (1.5, 1.5)  # Using plane for surface "table"
TABLE_LOCATION = (0, 0, -0.1)

# ========== CAMERA SETTINGS ==========
CAMERA_OBJECT_NAME = "RenderCamera"  # name given to the camera the sample is rendered from
CAMERA_LENS = 50  # Focal length in mm
CAMERA_SENSOR_WIDTH = 36  # Sensor width in mm

# Random tilt of the frame, since a handheld photograph is never perfectly square to the page.
CAMERA_ROLL_RANGE_DEG = (-8.0, 8.0)  # degrees, positive tilting clockwise

# Candidate view directions tried when looking for a valid camera angle: one ring of azimuths
# per inclination, so both lists below must have the same length. Widening these ranges gives
# more extreme viewpoints, at the cost of more candidates failing the visibility check.
# The default ranges stay deliberately close to a top-down, front-facing view.
CAMERA_INCLINATIONS_DEG = [5, 12, 20, 28]  # from the zenith: 0 = top-down, 90 = horizontal
CAMERA_AZIMUTH_RANGE_DEG = (-120, 120)  # 0 = reader position, +90 = right of the page
CAMERA_AZIMUTH_COUNTS = [1, 5, 9, 11]  # azimuths sampled at each inclination

# ========== LIGHTING SETTINGS ==========
# Ambient light from the world background, kept very dim so the lights below do the work and
# the shadows they cast stay readable.
WORLD_LIGHT_STRENGTH = 0.2
WORLD_LIGHT_COLOR = (0.01, 0.01, 0.01)

# Random rotation applied to the whole setup around the paper, on the Z axis. This is what
# keeps the light direction from repeating, since the setups below are fixed arrangements.
LIGHT_ROTATION_MIN_ANGLE = 0  # radians
LIGHT_ROTATION_MAX_ANGLE = 2 * 3.14159  # radians, a full turn

# The setups themselves are declared in lighting_setup.py, as LIGHTING_SETUPS; each of its
# entries points at one <PREFIX>_TYPE/_LOCATION/_ENERGY/_SIZE group below.
# Three-point lighting setup
KEY_LIGHT_TYPE = "AREA"
KEY_LIGHT_LOCATION = (-1.3, 1.1, 2.2)
KEY_LIGHT_ENERGY = 25
KEY_LIGHT_SIZE = 1.0

FILL_LIGHT_TYPE = "AREA"
FILL_LIGHT_LOCATION = (1.3, -0.17, 2.4)
FILL_LIGHT_ENERGY = 25
FILL_LIGHT_SIZE = 1.5

RIM_LIGHT_TYPE = "AREA"
RIM_LIGHT_LOCATION = (0, 2.5, 0.25)
RIM_LIGHT_ENERGY = 30
RIM_LIGHT_SIZE = 0.8

# Color temperature preset ranges
COLOR_TEMP_WARM_RANGE = ((1.0, 0.90, 0.70), (1.0, 0.95, 0.85))
COLOR_TEMP_COOL_RANGE = ((0.75, 0.85, 1.0), (0.90, 0.95, 1.0))
COLOR_TEMP_NEUTRAL_RANGE = ((0.98, 0.98, 0.98), (1.0, 1.0, 1.0))

# Rim lighting setup (edge/crease emphasis)
RIM_SETUP_LEFT_TYPE = "AREA"
RIM_SETUP_LEFT_LOCATION = (-2.0, 1.5, 0.8)
RIM_SETUP_LEFT_ENERGY = 40
RIM_SETUP_LEFT_SIZE = 1.0

RIM_SETUP_RIGHT_TYPE = "AREA"
RIM_SETUP_RIGHT_LOCATION = (2.0, 1.5, 0.8)
RIM_SETUP_RIGHT_ENERGY = 40
RIM_SETUP_RIGHT_SIZE = 1.0

RIM_SETUP_FILL_TYPE = "AREA"
RIM_SETUP_FILL_LOCATION = (0, -2.0, 1.5)
RIM_SETUP_FILL_ENERGY = 10
RIM_SETUP_FILL_SIZE = 1.5

# Softbox/studio lighting setup (even illumination)
SOFTBOX_TOP_TYPE = "AREA"
SOFTBOX_TOP_LOCATION = (0, 0, 3.0)
SOFTBOX_TOP_ENERGY = 30
SOFTBOX_TOP_SIZE = 2.0

SOFTBOX_LEFT_TYPE = "AREA"
SOFTBOX_LEFT_LOCATION = (-2.5, 0, 1.5)
SOFTBOX_LEFT_ENERGY = 25
SOFTBOX_LEFT_SIZE = 2.0

SOFTBOX_RIGHT_TYPE = "AREA"
SOFTBOX_RIGHT_LOCATION = (2.5, 0, 1.5)
SOFTBOX_RIGHT_ENERGY = 25
SOFTBOX_RIGHT_SIZE = 2.0

# Natural/window lighting setup (daylight simulation)
NATURAL_SUN_TYPE = "AREA"
NATURAL_SUN_LOCATION = (2.0, -2.0, 3.0)
NATURAL_SUN_ENERGY = 50
NATURAL_SUN_SIZE = 1.5

NATURAL_SKY_TYPE = "AREA"
NATURAL_SKY_LOCATION = (0, 0, 4.0)
NATURAL_SKY_ENERGY = 15
NATURAL_SKY_SIZE = 3.0

NATURAL_BOUNCE_TYPE = "AREA"
NATURAL_BOUNCE_LOCATION = (-1.5, 1.5, 0.5)
NATURAL_BOUNCE_ENERGY = 8
NATURAL_BOUNCE_SIZE = 1.8

# ========== PHYSICS SETTINGS ==========
# Rigid body simulation settling the table against the paper. Gravity points up so the table
# rises to meet the paper, which must not move: see plane_paper_contact.py.
SIMULATION_FRAMES = 250  # frames played through before the result is baked
SIMULATION_GRAVITY = (0.0, 0.0, 9.8100004196167)  # m/s^2, +Z so the table falls upward
RIGID_BODY_FRICTION = 100.0  # far above any physical value, so surfaces grip instead of sliding

# ========== MATERIAL SETTINGS ==========
BASE_PAPER_MATERIAL_BLEND = "./materials/paperTexture.blend"
BASE_PAPER_MATERIAL_NAME = "Paper"

# ========== DEBUG SETTINGS ==========
VERBOSE = False  # print debug messages
