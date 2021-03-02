import numpy as np
from tensorflow.keras.backend import count_params


def normalize_value(v):
    if isinstance(
        v,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(v)

    elif isinstance(v, (np.float_, np.float16, np.float32, np.float64)):
        return float(v)

    elif isinstance(v, (np.complex_, np.complex64, np.complex128)):
        return {"real": v.real, "imag": v.imag}

    elif isinstance(v, (np.ndarray,)):
        return v.tolist()

    elif isinstance(v, (np.bool_)):
        return bool(v)

    elif isinstance(v, (np.void)):
        return None

    return v


def conform_dict(d):
    """Normalize a dictionary for JSON serialization, casting numpy types

    Args:
        d (dict): Input dictionary

    Returns:
        dict: Normalized dictionary
    """
    if type(d) != dict:
        return normalize_value(d)
    for k, v in d.items():
        if type(v) == dict:
            d[k] = conform_dict(v)
        elif type(v) == list:
            d[k] = [conform_dict(x) for x in v]
        else:
            d[k] = normalize_value(v)
    return d


def get_summary(model):
    summary_list = []
    model.summary(print_fn=lambda s: summary_list.append(s))
    return "\n".join(summary_list)


def count_model_params(model):
    trainable_count = int(sum([count_params(p) for p in model.trainable_weights]))
    non_trainable_count = int(
        sum([count_params(p) for p in model.non_trainable_weights])
    )
    return {
        "total": trainable_count + non_trainable_count,
        "trainable": trainable_count,
        "non_trainable": non_trainable_count,
    }


def get_layers_summary(model):
    # save model configuration (we can do better)
    model_summary = []
    for li, l in enumerate(model.layers):
        model_summary.append(
            {
                "layer_n": li,
                "name": l.name,
                "input_shape": l.input_shape,
                "output_shape": l.output_shape,
            }
        )
    return model_summary


def get_model_info(model, source, loss=None):
    """Get information about a Keras Model

    Args:
        model: an instance of `keras.Model`
        source (string): The source framework (only `keras` is currently supported)
        loss (string or loss function, optional): Loss function used for training.
            Defaults to None.

    Raises:
        Exception: When another source than `keras` is used

    Returns:
        dict: Keras model information
    """
    if source == "keras":
        if loss is None and hasattr(model, "loss"):
            loss = model.loss if type(model.loss) == str else model.loss.name
        return {
            "name": model.name,
            "param_count": count_model_params(model),
            "summary": get_summary(model),
            "layers": get_layers_summary(model),
            "loss": loss or "Unknown",
        }
    else:
        raise Exception("Only 'keras' source is implemented at the moment")
