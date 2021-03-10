

def save_model(individual, epoch, out_dir):

    model = individual.to_phenotype()
    model.save(f'{out_dir}/checkpoints/epoch-{epoch}/model')
    del model


def save_experiment_description(hms, exp_name, seed, out_dir):

    with open(f'{out_dir}/description.txt', 'w') as file:
        file.write(f'Experiment: {exp_name}\nInitial seed: {seed}\n')
        file.write(str(hms))
