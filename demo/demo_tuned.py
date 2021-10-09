import sys
import json
sys.path.append(".")

import tqdm, os
def nop(it, *a, **k): return it
tqdm.tqdm = nop
os.environ.get("TQDM_CONFIG", '')

from transformers import logging
logging.set_verbosity(logging.CRITICAL)

from openprompt.utils.logging import logger
from openprompt.utils.utils import load_checkpoint

logger.setLevel(logging.CRITICAL)

import torch

def color(text, color="\033[35m"): # or \033[32m
    return color+text+"\033[0m"

def input_selection(what, lis, delimiter="\n"):
    print(f"Select a {color(what)}: ")
    for idx, item in enumerate(lis):
        print(f"    {idx+1}.", item, end=delimiter)
    if delimiter != '\n': print()
    idx = int(input(f"Enter a number between 1 to {len(lis)} :   "))-1
    print()
    return lis[idx]

def input_enter(what):
    res = input(f"Enter the {color(what)}: ")
    print()
    return res

def progress_print(text):
    print(text, "...")

if __name__ == "__main__":
    with torch.no_grad():
        from openprompt.plms import get_model_class
        model_class = get_model_class(plm_type = "roberta")
        model_path = "roberta-large"
        bertConfig = model_class.config.from_pretrained(model_path)
        bertTokenizer = model_class.tokenizer.from_pretrained(model_path)
        bertModel = model_class.model.from_pretrained(model_path)

        print()
        progress_print(f"This demo is powered by {color('OpenPrompt')}")
        print()

        while True:
            from openprompt.data_utils import InputExample
            text = input_enter("text")
            '''
            Albert Einstein was one of the greatest intellects of his time.
            '''
            dataset = [
                InputExample(
                    guid = 0,
                    text_a = text,
                )
            ]

            template = input_enter("Prompt Template")
            '''
                <text_a> It is <mask>
                <text_a> Albert Einstein is a <mask>
                Albert Einstein was born in <mask>
            '''
            from openprompt.prompts import ManualTemplate
            template = ManualTemplate(
                text = template.split(),
                tokenizer = bertTokenizer,
            )

            verbalizer = input_selection("Prompt Verbalizer", [
                'Sentiment Verbalizer',
                'Entity Verbalizer',
                'Knowledge Probing',
                'Customize'
            ])
            classes = None
            logging_path = None
            if verbalizer == 'Knowledge Probing':
                verbalizer = None
                classes = {v:k for k,v in bertTokenizer.get_vocab().items()}
                # logging_path = "logs/LAMA"
            else:
                label_words = None
                if verbalizer == "Entity Verbalizer":
                    label_words = {
                        "person-actor": ["actor"],
                        "person-director": ["director"],
                        "person-artist/author": ["artist", "author"],
                        "person-athlete": ["athlete"],
                        "person-politician": ["politician"],
                        "person-scholar": ["scholar", "scientist"],
                        "person-soldier": ["soldier"],
                        "person-other": ["person"],

                        "organization-showorganization": ["show", "organization"],
                        "organization-religion": ["religion"],
                        "organization-company": ["company"],
                        "organization-sportsteam": ["sports", "team"],
                        "organization-education": ["education"],
                        "organization-government/governmentagency": ["government", "agency"],
                        "organization-media/newspaper": ["media", "newspaper"],
                        "organization-politicalparty": ["political", "party"],
                        "organization-sportsleague": ["sports", "league"],
                        "organization-other": ["organization"],

                        "location-GPE": ["geopolitical"],
                        "location-road/railway/highway/transit": ["road", "railway", "highway", "transit"],
                        "location-bodiesofwater": ["water"],
                        "location-park": ["park"],
                        "location-mountain": ["mountain"],
                        "location-island": ["island"],
                        "location-other": ["location"],

                        "product-software": ["software"],
                        "product-food": ["food"],
                        "product-game": ["game"],
                        "product-ship": ["ship"],
                        "product-train": ["train"],
                        "product-airplane": ["airplane"],
                        "product-car": ["car"],
                        "product-weapon": ["weapon"],
                        "product-other": ["product"],

                        "building-theater": ["theater"],
                        "building-sportsfacility": ["sports", "facility"],
                        "building-airport": ["airport"],
                        "building-hospital": ["hospital"],
                        "building-library": ["library"],
                        "building-hotel": ["hotel"],
                        "building-restaurant": ["restaurant"],
                        "building-other": ["building"],

                        "event-sportsevent": ["sports", "event"],
                        "event-attack/battle/war/militaryconflict": ["attack", "battle", "war", "military", "conflict"],
                        "event-disaster": ["disaster"],
                        "event-election": ["election"],
                        "event-protest": ["protest"],
                        "event-other": ["event"],

                        "art-music": ["music"],
                        "art-writtenart": ["written", "art"],
                        "art-film": ["film"],
                        "art-painting": ["painting"],
                        "art-broadcastprogram": ["broadcast", "program"],
                        "art-other": ["art"],

                        "other-biologything": ["biology"],
                        "other-chemicalthing": ["chemical"],
                        "other-livingthing": ["living"],
                        "other-astronomything": ["astronomy"],
                        "other-god": ["god"],
                        "other-law": ["law"],
                        "other-award": ["award"],
                        "other-disease": ["disease"],
                        "other-medical": ["medical"],
                        "other-language": ["language"],
                        "other-currency": ["currency"],
                        "other-educationaldegree": ["educational", "degree"]
                    }
                    logging_path = "logs/FewNERD"
                elif verbalizer == "Sentiment Verbalizer":
                    label_words = {
                        "negative": ["abysmal", "adverse", "alarming", "angry", "annoy", "anxious", "apathy", "appalling", "atrocious", "awful", "bad", "banal", "barbed", "belligerent", "bemoan", "beneath", "boring", "broken", "callous", "can't", "clumsy", "coarse", "cold", "cold-hearted", "collapse", "confused", "contradictory", "contrary", "corrosive", "corrupt", "crazy", "creepy", "criminal", "cruel", "cry", "cutting", "damage", "damaging", "dastardly", "dead", "decaying", "deformed", "deny", "deplorable", "depressed", "deprived", "despicable", "detrimental", "dirty", "disease", "disgusting", "disheveled", "dishonest", "dishonorable", "dismal", "distress", "don't", "dreadful", "dreary", "enraged", "eroding", "evil", "fail", "faulty", "fear", "feeble", "fight", "filthy", "foul", "frighten", "frightful", "gawky", "ghastly", "grave", "greed", "grim", "grimace", "gross", "grotesque", "gruesome", "guilty", "haggard", "hard", "hard-hearted", "harmful", "hate", "hideous", "homely", "horrendous", "horrible", "hostile", "hurt", "hurtful", "icky", "ignorant", "ignore", "ill", "immature", "imperfect", "impossible", "inane", "inelegant", "infernal", "injure", "injurious", "insane", "insidious", "insipid", "jealous", "junky", "lose", "lousy", "lumpy", "malicious", "mean", "menacing", "messy", "misshapen", "missing", "misunderstood", "moan", "moldy", "monstrous", "naive", "nasty", "naughty", "negate", "negative", "never", "no", "nobody", "nondescript", "nonsense", "not", "noxious", "objectionable", "odious", "offensive", "old", "oppressive", "pain", "perturb", "pessimistic", "petty", "plain", "poisonous", "poor", "prejudice", "questionable", "quirky", "quit", "reject", "renege", "repellant", "reptilian", "repugnant", "repulsive", "revenge", "revolting", "rocky", "rotten", "rude", "ruthless", "sad", "savage", "scare", "scary", "scream", "severe", "shocking", "shoddy", "sick", "sickening", "sinister", "slimy", "smelly", "sobbing", "sorry", "spiteful", "sticky", "stinky", "stormy", "stressful", "stuck", "stupid", "substandard", "suspect", "suspicious", "tense", "terrible", "terrifying", "threatening", "ugly", "undermine", "unfair", "unfavorable", "unhappy", "unhealthy", "unjust", "unlucky", "unpleasant", "unsatisfactory", "unsightly", "untoward", "unwanted", "unwelcome", "unwholesome", "unwieldy", "unwise", "upset", "vice", "vicious", "vile", "villainous", "vindictive", "wary", "weary", "wicked", "woeful", "worthless", "wound", "yell", "yucky", "zero"],
                        "positive": ["absolutely", "accepted", "acclaimed", "accomplish", "accomplishment", "achievement", "action", "active", "admire", "adorable", "adventure", "affirmative", "affluent", "agree", "agreeable", "amazing", "angelic", "appealing", "approve", "aptitude", "attractive", "awesome", "beaming", "beautiful", "believe", "beneficial", "bliss", "bountiful", "bounty", "brave", "bravo", "brilliant", "bubbly", "calm", "celebrated", "certain", "champ", "champion", "charming", "cheery", "choice", "classic", "classical", "clean", "commend", "composed", "congratulation", "constant", "cool", "courageous", "creative", "cute", "dazzling", "delight", "delightful", "distinguished", "divine", "earnest", "easy", "ecstatic", "effective", "effervescent", "efficient", "effortless", "electrifying", "elegant", "enchanting", "encouraging", "endorsed", "energetic", "energized", "engaging", "enthusiastic", "essential", "esteemed", "ethical", "excellent", "exciting", "exquisite", "fabulous", "fair", "familiar", "famous", "fantastic", "favorable", "fetching", "fine", "fitting", "flourishing", "fortunate", "free", "fresh", "friendly", "fun", "funny", "generous", "genius", "genuine", "giving", "glamorous", "glowing", "good", "gorgeous", "graceful", "great", "green", "grin", "growing", "handsome", "happy", "harmonious", "healing", "healthy", "hearty", "heavenly", "honest", "honorable", "honored", "hug", "idea", "ideal", "imaginative", "imagine", "impressive", "independent", "innovate", "innovative", "instant", "instantaneous", "instinctive", "intellectual", "intelligent", "intuitive", "inventive", "jovial", "joy", "jubilant", "keen", "kind", "knowing", "knowledgeable", "laugh", "learned", "legendary", "light", "lively", "lovely", "lucid", "lucky", "luminous", "marvelous", "masterful", "meaningful", "merit", "meritorious", "miraculous", "motivating", "moving", "natural", "nice", "novel", "now", "nurturing", "nutritious", "okay", "one", "one-hundred percent", "open", "optimistic", "paradise", "perfect", "phenomenal", "pleasant", "pleasurable", "plentiful", "poised", "polished", "popular", "positive", "powerful", "prepared", "pretty", "principled", "productive", "progress", "prominent", "protected", "proud", "quality", "quick", "quiet", "ready", "reassuring", "refined", "refreshing", "rejoice", "reliable", "remarkable", "resounding", "respected", "restored", "reward", "rewarding", "right", "robust", "safe", "satisfactory", "secure", "seemly", "simple", "skilled", "skillful", "smile", "soulful", "sparkling", "special", "spirited", "spiritual", "stirring", "stunning", "stupendous", "success", "successful", "sunny", "super", "superb", "supporting", "surprising", "terrific", "thorough", "thrilling", "thriving", "tops", "tranquil", "transformative", "transforming", "trusting", "truthful", "unreal", "unwavering", "up", "upbeat", "upright", "up", "standing", "valued", "vibrant", "victorious", "victory", "vigorous", "virtuous", "vital", "vivacious", "wealthy", "welcome", "well", "whole", "wholesome", "willing", "wonderful", "wondrous", "worthy", "wow", "yes", "yummy", "zeal", "zealous"]
                    }
                    logging_path = "logs/IMDB"
                elif verbalizer == "Customize":
                    label_words = json.loads(input(">>> "))
                classes = list(label_words.keys())
                from openprompt.prompts import ManualVerbalizer
                verbalizer = ManualVerbalizer(
                    classes = classes,
                    label_words = label_words,
                    tokenizer = bertTokenizer
                )

            progress_print(f"Incorporating {color('Template')} and {color('Verbalizer')} into a {color('PromptModel')}")
            from openprompt import PromptForClassification
            prompt_model = PromptForClassification(
                template = template,
                model = bertModel,
                verbalizer = verbalizer,
            )

            if logging_path:
                state_dict = load_checkpoint(
                    load_path=logging_path,
                    load_best = True,
                    map_location="cpu", # cpu to prevent CUDA out of memory.
                )
                
                # load state to model
                prompt_model.load_state_dict(state_dict['state_dict'])
            
            prompt_model = prompt_model.to("cuda:0")

            progress_print("Predicting")
            print()

            from openprompt import PromptDataLoader
            data_loader = PromptDataLoader(
                dataset = dataset,
                tokenizer = bertTokenizer, 
                template = template, 
            )

            prompt_model.eval()
            for batch in data_loader:
                batch = batch.to("cuda:0").to_dict()
                if verbalizer is None:
                    logits = prompt_model.forward_without_verbalize(batch)
                else:
                    logits = prompt_model(batch)
                pred = torch.argmax(logits, dim=-1)
                pred = pred.cpu().tolist()

            if verbalizer == None:
                print(f"{color('Predicition')}: ", bertTokenizer.convert_tokens_to_string(bertTokenizer.convert_ids_to_tokens(pred)[0]))
            else:
                print(f"{color('Predicition')}: ", classes[pred[0]], f"(triggered by label words: {label_words[classes[pred[0]]][:6]})")

            print()
            print("=======================================================")
            print()
