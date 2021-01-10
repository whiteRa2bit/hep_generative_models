from generation.metrics.physical.utils import transform_signals
from generation.metrics.physical.energy_metrics import get_energy_metrics_dict
from generation.metrics.physical.space_metrics import get_space_metrics_dict
from generation.metrics.physical.time_metrics import get_time_metrics_dict


def get_physical_metrics_dict(real_signals_tensor, fake_signals_tensor):
    real_signals = transform_signals(real_signals_tensor)
    fake_signals = transform_signals(fake_signals_tensor)
    energy_metrics_dict = get_energy_metrics_dict(real_signals, fake_signals)
    space_metrics_dict = get_space_metrics_dict(real_signals, fake_signals)
    time_metrics_dict = get_time_metrics_dict(real_signals, fake_signals)
    physical_metrics_dict = {**energy_metrics_dict, **space_metrics_dict, **time_metrics_dict}
    return physical_metrics_dict
