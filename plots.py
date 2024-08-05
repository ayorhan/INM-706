import matplotlib.pyplot as plt

# Actual data
learning_rates = [0.1, 0.01, 0.001, 0.0001]
embed_dims = [32, 64, 128, 256]
hidden_dims = [32, 64, 128]
batch_sizes = [16, 32, 64, 128]

# Results for learning rate
lr_metrics = {
    'accuracy': {
        'train': [0.8907, 0.9243, 0.8918, 0.6726],
        'val': [0.8220, 0.8660, 0.8604, 0.6640],
        'test': [0.8133, 0.8444, 0.8587, 0.6732]
    },
    'loss': {
        'train': [0.0046, 0.0030, 0.0042, 0.0100],
        'val': [0.0068, 0.0061, 0.0052, 0.0100],
        'test': [0.0068, 0.0072, 0.0051, 0.0098]
    },
    'precision': {
        'train': [0.85772, 0.9241, 0.88771, 0.66099],
        'val': [0.7594, 0.88145, 0.84591, 0.64958],
        'test': [0.74739, 0.87612, 0.85311, 0.65709]
    },
    'recall': {
        'train': [0.93512, 0.92521, 0.89822, 0.70986],
        'val': [0.95467, 0.84059, 0.87423, 0.70695],
        'test': [0.94656, 0.80232, 0.86656, 0.72448]
    },
    'f1_score': {
        'train': [0.89475, 0.92465, 0.89293, 0.68456],
        'val': [0.84591, 0.86053, 0.85984, 0.67705],
        'test': [0.83527, 0.8376, 0.85978, 0.68914]
    },
    'auc': {
        'train': [0.89091, 0.9243, 0.89172, 0.67257],
        'val': [0.81879, 0.86559, 0.86068, 0.66415],
        'test': [0.81332, 0.84444, 0.85868, 0.6732]
    }
}

# Results for embed dimension
embed_metrics = {
    'accuracy': {
        'train': [0.9312, 0.9321, 0.9350, 0.9345],
        'val': [0.8556, 0.8552, 0.8480, 0.8564],
        'test': [0.8510, 0.8400, 0.8415, 0.8396]
    },
    'loss': {
        'train': [0.0030, 0.0028, 0.0026, 0.0027],
        'val': [0.0059, 0.0064, 0.0078, 0.0065],
        'test': [0.0061, 0.0068, 0.0078, 0.0071]
    },
    'precision': {
        'train': [0.93058, 0.93325, 0.93266, 0.93487],
        'val': [0.85393, 0.86038, 0.82015, 0.88913],
        'test': [0.85736, 0.85248, 0.8238, 0.87654]
    },
    'recall': {
        'train': [0.93252, 0.93025, 0.93769, 0.93346],
        'val': [0.85114, 0.85292, 0.89164, 0.81901],
        'test': [0.842, 0.8224, 0.86888, 0.79064]
    },
    'f1_score': {
        'train': [0.93155, 0.93175, 0.93517, 0.93416],
        'val': [0.85253, 0.85663, 0.85441, 0.85263],
        'test': [0.84961, 0.83717, 0.84574, 0.83138]
    },
    'auc': {
        'train': [0.93114, 0.93209, 0.935, 0.93445],
        'val': [0.85552, 0.85523, 0.84798, 0.85695],
        'test': [0.85096, 0.84004, 0.84152, 0.83964]
    }
}

# Results for hidden dimension
hidden_metrics = {
    'accuracy': {
        'train': [0.9277, 0.9226, 0.9281],
        'val': [0.8602, 0.8674, 0.8504],
        'test': [0.8498, 0.8528, 0.8372]
    },
    'loss': {
        'train': [0.0031, 0.0032, 0.0030],
        'val': [0.0058, 0.0054, 0.0060],
        'test': [0.0062, 0.0060, 0.0063]
    },
    'precision': {
        'train': [0.93098, 0.92298, 0.92835],
        'val': [0.832, 0.87112, 0.88298],
        'test': [0.82615, 0.859, 0.88613]
    },
    'recall': {
        'train': [0.92351, 0.92205, 0.92807],
        'val': [0.90614, 0.86313, 0.80452],
        'test': [0.88616, 0.84416, 0.77384]
    },
    'f1_score': {
        'train': [0.92723, 0.92251, 0.92821],
        'val': [0.86749, 0.86711, 0.84193],
        'test': [0.8551, 0.85152, 0.82619]
    },
    'auc': {
        'train': [0.92769, 0.9226, 0.92805],
        'val': [0.85974, 0.86741, 0.84996],
        'test': [0.84984, 0.8528, 0.8372]
    }
}

# Results for batch size
batch_metrics = {
    'accuracy': {
        'train': [0.9451, 0.9266, 0.9252, 0.9279],
        'val': [0.8490, 0.8558, 0.8550, 0.8542],
        'test': [0.8358, 0.8453, 0.8446, 0.8359]
    },
    'loss': {
        'train': [0.0090, 0.0061, 0.0032, 0.0015],
        'val': [0.0438, 0.0120, 0.0057, 0.0032],
        'test': [0.0122, 0.0061, 0.0062, 0.0066]
    },
    'precision': {
        'train': [0.9452, 0.93006, 0.92466, 0.92863],
        'val': [0.8127, 0.87034, 0.87825, 0.88225],
        'test': [0.79784, 0.86985, 0.87505, 0.87338]
    },
    'recall': {
        'train': [0.94501, 0.92243, 0.9254, 0.92733],
        'val': [0.90673, 0.83739, 0.82786, 0.81507],
        'test': [0.89952, 0.81216, 0.804, 0.78576]
    },
    'f1_score': {
        'train': [0.94511, 0.92623, 0.92503, 0.92798],
        'val': [0.85714, 0.85354, 0.85231, 0.84733],
        'test': [0.84564, 0.84001, 0.83802, 0.82726]
    },
    'auc': {
        'train': [0.9451, 0.9266, 0.9252, 0.9279],
        'val': [0.84905, 0.85587, 0.8553, 0.85392],
        'test': [0.8358, 0.84532, 0.8446, 0.83592]
    }
}

def plot_metrics(parameter_values, parameter_name, metrics):
    for metric_name, metric_values in metrics.items():
        plt.figure()
        plt.plot(parameter_values, metric_values['train'], label='Train')
        plt.plot(parameter_values, metric_values['val'], label='Validation')
        plt.plot(parameter_values, metric_values['test'], label='Test')
        plt.xlabel(parameter_name)
        plt.ylabel(metric_name.capitalize())
        plt.title(f'{metric_name.capitalize()} vs {parameter_name.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Plot graphs for learning rates
plot_metrics(learning_rates, 'Learning Rate', lr_metrics)

# Plot graphs for embed dimensions
plot_metrics(embed_dims, 'Embed Dimension', embed_metrics)

# Plot graphs for hidden dimensions
plot_metrics(hidden_dims, 'Hidden Dimension', hidden_metrics)

# Plot graphs for batch sizes
plot_metrics(batch_sizes, 'Batch Size', batch_metrics)
