import torch
from ldm.systems.acrobot.utils.observation_preprocesor import AcrobotObservationPreprocessor
from ldm.systems.car.dynamics.decoupled_neural_tire_model import DecoupledNeuralTireModel
from ldm.systems.car.dynamics.decoupled_pacejka_tire_model import DecoupledPacejkaTireModel
from ldm.systems.car.dynamics.dugoff_tire_model import DugoffTireModel
from ldm.systems.car.dynamics.dummy_kicajka_tire_model import DummyKicajkaTireModel
from ldm.systems.car.dynamics.exp_tanh_tire_model import ExpTanhTireModel
from ldm.systems.car.dynamics.fiala_tire_model import FialaTireModel
from ldm.systems.car.dynamics.kicajka23_tire_model import Kicajka23TireModel
from ldm.systems.car.dynamics.kicajka2_tire_model import Kicajka2TireModel
from ldm.systems.car.dynamics.kicajka3_tire_model import Kicajka3TireModel
from ldm.systems.car.dynamics.kicajka4_tire_model import Kicajka4TireModel
from ldm.systems.car.dynamics.kicajka5_tire_model import Kicajka5TireModel
from ldm.systems.car.dynamics.kicajka_tire_model import KicajkaTireModel
from ldm.systems.car.dynamics.linear_tire_model import LinearTireModel
from ldm.systems.car.dynamics.load_transfer_pacejka_tire_model import LoadTransferPacejkaTireModel
from ldm.systems.car.dynamics.neural_exp_tanh_tire_model import NeuralExpTanhTireModel
from ldm.systems.car.dynamics.neural_kicajka4_tire_model import NeuralKicajka4TireModel
from ldm.systems.car.dynamics.neural_tire_model import NeuralTireModel
from ldm.systems.car.dynamics.neural_tire_residual_pacejka_model import NeuralTireResidualPacejkaModel
from ldm.systems.car.dynamics.pacejka_friction_ellipse_tire_model import PacejkaFrictionEllipseTireModel
from ldm.systems.car.dynamics.pacejka_mf52_tire_model import PacejkaMF52TireModel
from ldm.systems.car.dynamics.pacejka_mf61_tire_model import PacejkaMF61TireModel
from ldm.systems.car.dynamics.pacejka_offset_tire_model import PacejkaOffsetTireModel
from ldm.systems.car.dynamics.pacejka_tire_model import PacejkaTireModel
from ldm.systems.car.dynamics.single_track import SingleTrack
from ldm.systems.car.dynamics.single_track_load_transfer import SingleTrackLoadTransfer
from ldm.systems.car.dynamics.tmeasy_tire_model import TMeasyTireModel
from ldm.systems.commons.residual_model import ResidualModel
from ldm.systems.f1tenth.utils.f1tenth_observation_preprocesor import CarObservationPreprocessor
from ldm.systems.vector_field.utils.dummy_vf_preprocessor import DummyVFPreprocessor
from ldm.systems.vw_golf.dynamics.single_track import VWGolfSingleTrack
from ldm.systems.vw_golf.utils.golf_observation_preprocesor import GolfObservationPreprocessor
from ldm.utils.state_extender import StateExtender
from ldm.systems.mlp.mlp import Mlp
from ldm.systems.mlp.mlp_with_history import MlpWithHistory


def get_dynamics_model(system: str, model_type: str, state_extender_type: str = None, *args, **kwargs):
    # Environment specific settings
    if system == "f1tenth":
        base_model_input_dim = 3  # v_x, v_y, r
        preprocessor = CarObservationPreprocessor()
        #layer_sizes = [5, 128, 128, 3]
        layer_sizes = [6, 32, 32, 3]
        activation = torch.nn.ReLU()
    elif system == "vw_golf":
        base_model_input_dim = 3  # v_x, v_y, r
        preprocessor = GolfObservationPreprocessor()
        layer_sizes = [6, 32, 32, 3]
        activation = torch.nn.ReLU()
    elif system == "acrobot":
        preprocessor = AcrobotObservationPreprocessor()
        layer_sizes = [7, 32, 32, 4]
        activation = torch.nn.ReLU()
    elif system == "vector_field":
        preprocessor = DummyVFPreprocessor()
        #layer_sizes = [2, 32, 32, 2]
        layer_sizes = [2, 16, 16, 2]
        activation = torch.nn.ReLU()
        base_model_input_dim = 2
    else:
        raise ValueError(f"Unknown system: {system}")

    if "nn" in kwargs:
        layer_sizes = layer_sizes[:1] + [kwargs["nn"]] * (len(layer_sizes) - 2) + layer_sizes[-1:]

    # Initialize state extender if needed
    if state_extender_type is not None:
        state_extender_type, state_extender_output_dim = state_extender_type.split("-")
        state_extender_output_dim = int(state_extender_output_dim)
        state_extender = StateExtender(state_extender_type,
                                        input_dim=layer_sizes[0],
                                        output_dim=state_extender_output_dim,
                                        preprocessor=preprocessor)
        layer_sizes[0] += state_extender_output_dim
        layer_sizes[-1] += state_extender_output_dim
    else:
        state_extender = None

    if system == "f1tenth":
        if "single_track_load_transfer" in model_type:
            from ldm.systems.f1tenth.dynamics.single_track_load_transfer_params import F1TenthSingleTrackLoadTransferParameters
            vehicle_params = F1TenthSingleTrackLoadTransferParameters()
        elif "single_track" in model_type:
            from ldm.systems.f1tenth.dynamics.single_track_params import F1TenthSingleTrackParameters
            vehicle_params = F1TenthSingleTrackParameters()
    elif system == "vw_golf":
        from ldm.systems.vw_golf.dynamics.single_track_params import VWGolfSingleTrackParameters
        vehicle_params = VWGolfSingleTrackParameters()


    residual = model_type.startswith("residual_")
    if residual:
        model_type = model_type[len("residual_"):]
    model = None

    if model_type == "pacejka_single_track":
        tire_model = PacejkaTireModel(*args, **kwargs)
    elif model_type == "decoupled_pacejka_single_track":
        tire_model = DecoupledPacejkaTireModel(*args, **kwargs)
    elif model_type == "pacejka_offset_single_track":
        tire_model = PacejkaOffsetTireModel(*args, **kwargs)
    elif model_type == "pacejka_friction_ellipse_single_track":
        tire_model = PacejkaFrictionEllipseTireModel(*args, **kwargs)
    elif model_type == "pacejka_mf52_single_track":
        tire_model = PacejkaMF52TireModel(*args, **kwargs)
    elif model_type == "pacejka_mf61_single_track":
        tire_model = PacejkaMF61TireModel(*args, **kwargs)
    elif model_type == "dummy_kicajka_single_track":
        tire_model = DummyKicajkaTireModel(*args, **kwargs)
    elif model_type == "kicajka_single_track":
        tire_model = KicajkaTireModel(*args, **kwargs)
    elif model_type == "kicajka2_single_track":
        tire_model = Kicajka2TireModel(*args, **kwargs)
    elif model_type == "kicajka3_single_track":
        tire_model = Kicajka3TireModel(*args, **kwargs)
    elif model_type == "kicajka4_single_track":
        tire_model = Kicajka4TireModel(*args, **kwargs)
    elif model_type == "kicajka5_single_track":
        tire_model = Kicajka5TireModel(*args, **kwargs)
    elif model_type == "kicajka23_single_track":
        tire_model = Kicajka23TireModel(*args, **kwargs)
    elif model_type == "neural_kicajka4_single_track":
        tire_model = NeuralKicajka4TireModel(input_type="vinvariant", *args, **kwargs)
    elif model_type == "neural_kicajka4_single_track_state":
        tire_model = NeuralKicajka4TireModel(n_in=6, input_type="state", *args, **kwargs)
    elif model_type == "pacejka_single_track_load_transfer":
        tire_model = LoadTransferPacejkaTireModel()
        base_model_input_dim += 1 # due to dFz input
    elif model_type == "neural_single_track":
        tire_model = NeuralTireModel(input_type="vinvariant", *args, **kwargs)
    elif model_type == "neural_single_track_state":
        tire_model = NeuralTireModel(n_in=6, input_type="state", *args, **kwargs)
    elif model_type == "neural_single_track_load_transfer":
        tire_model = NeuralTireModel(input_type="vinvariant", *args, **kwargs)
        base_model_input_dim += 1 # due to dFz input
    elif model_type == "decoupled_neural_single_track":
        tire_model = DecoupledNeuralTireModel(*args, **kwargs)
    elif model_type == "decoupled_neural_single_track_load_transfer":
        tire_model = DecoupledNeuralTireModel(*args, **kwargs)
        base_model_input_dim += 1 # due to dFz input
    elif model_type == "decoupled_neural_residual_pacejka_single_track":
        tire_model = NeuralTireResidualPacejkaModel(n_in=5, input_type="vinvariant", *args, **kwargs)
    elif model_type == "neural_exptanh_single_track":
        tire_model = NeuralExpTanhTireModel(*args, **kwargs)
    elif model_type == "exptanh_single_track":
        tire_model = ExpTanhTireModel(*args, **kwargs)
    elif model_type == "tmeasy_single_track":
        tire_model = TMeasyTireModel(*args, **kwargs)
    elif model_type =="fiala_single_track":
        tire_model = FialaTireModel(*args, **kwargs)
    elif model_type =="dugoff_single_track":
        tire_model = DugoffTireModel(*args, **kwargs)
    elif model_type =="linear_single_track":
        tire_model = LinearTireModel(*args, **kwargs)
    elif model_type == "mlp":
        model = Mlp(preprocessor=preprocessor, layer_sizes=layer_sizes, activation=activation)
    elif model_type == "mlp_history":
        history_len = kwargs.get("history_len", 10)
        obs_dim = layer_sizes[0]  # obs_dim matches base input dim (before history augmentation)
        model = MlpWithHistory(
            preprocessor=preprocessor,
            layer_sizes=layer_sizes,
            activation=activation,
            history_len=history_len,
            obs_dim=obs_dim,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if "single_track_load_transfer" in model_type:
        model = SingleTrackLoadTransfer(vehicle_parameters=vehicle_params, tire_model=tire_model)
    elif "single_track" in model_type:
        if system == "f1tenth":
            model = SingleTrack(vehicle_parameters=vehicle_params, tire_model=tire_model)
        elif system == "vw_golf":
            model = VWGolfSingleTrack(vehicle_parameters=vehicle_params, tire_model=tire_model)
        else:
            raise ValueError(f"Unknown system for single track model: {system}")


    if residual:
        model = ResidualModel(base_model=model,
                              base_model_input_dim=base_model_input_dim,
                              preprocessor=preprocessor,
                              layer_sizes=layer_sizes,
                              activation=activation)

    return model, state_extender