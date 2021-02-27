from . import DIR


def save_model(individual, epoch):

    model = individual.to_phenotype()
    model.save(f'{DIR}/checkpoints/epoch-{epoch}/model')
    del model


def save_experiment_description(hms, exp_name, seed):

    with open(f'{DIR}/description.txt', 'w') as file:
        file.write(f'Experiment: {exp_name}\nInitial seed: {seed}\n')
        file.write(str(hms))
