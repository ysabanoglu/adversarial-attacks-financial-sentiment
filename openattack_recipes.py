import OpenAttack
from datasets import load_from_disk
from datasets import Dataset
import openattack_recipes as init


def dataset_for_recipes(dataset):

    dataset = load_from_disk(dataset)
    df_pd = dataset.to_pandas()
    df_pd['Label'] = df_pd['Label'].astype(int)
    df_pd = df_pd[['Sentence_ID', 'Sentence', 'Label']]
    df_pd = df_pd.rename(columns={'Sentence': 'x', 'Label': 'y'})
    dataset = Dataset.from_pandas(df_pd)
    return dataset


def initialize_recipe(recipe_code):

    attacker_initializers = {
        'BAE': init.initialize_bae_attacker,
        'BERT': init.initialize_bert_attacker,
        'DWB': init.initialize_deep_word_bug_attacker,
        'FD': init.initialize_fd_attacker,
        'GAN': init.initialize_gan_attacker,
        'GEO': init.initialize_geo_attacker,
        'GEN': init.initialize_genetic_attacker,
        'HOT': init.initialize_hot_flip_attacker,
        'PSO': init.initialize_pso_attacker,
        'PWWS': init.initialize_pwws_attacker,
        'SCPN': init.initialize_scpn_attacker,
        'TBG': init.initialize_text_bugger_attacker,
        'TF': init.initialize_text_fooler_attacker,
        'UAT': init.initialize_uat_attacker,
        'VIP': init.initialize_viper_attacker,
    }

    initializer = attacker_initializers.get(recipe_code)
    if initializer:

        return initializer()
    else:
        print(f"Attacker code '{recipe_code}' not recognized.")

        return None


def initialize_bae_attacker():
    return OpenAttack.attackers.BAEAttacker()


def initialize_bert_attacker():
    return OpenAttack.attackers.BERTAttacker()


def initialize_deep_word_bug_attacker():
    return OpenAttack.attackers.DeepWordBugAttacker()


def initialize_fd_attacker():
    return OpenAttack.attackers.FDAttacker()


def initialize_gan_attacker():
    return OpenAttack.attackers.GANAttacker()


def initialize_geo_attacker():
    return OpenAttack.attackers.GEOAttacker()


def initialize_genetic_attacker():
    return OpenAttack.attackers.GeneticAttacker()


def initialize_hot_flip_attacker():
    return OpenAttack.attackers.HotFlipAttacker()


def initialize_pso_attacker():
    return OpenAttack.attackers.PSOAttacker()


def initialize_pwws_attacker():
    return OpenAttack.attackers.PWWSAttacker()


def initialize_scpn_attacker():
    return OpenAttack.attackers.SCPNAttacker()


def initialize_text_bugger_attacker():
    return OpenAttack.attackers.TextBuggerAttacker()


def initialize_text_fooler_attacker():
    return OpenAttack.attackers.TextFoolerAttacker()


def initialize_uat_attacker():
    return OpenAttack.attackers.UATAttacker()


def initialize_viper_attacker():
    return OpenAttack.attackers.VIPERAttacker()
