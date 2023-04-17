import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """

        assert "<sent> " in input_text and " </sent>" in input_text, "Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side."
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        final_input_text = f"lexical = {lex_code}, order = {order_code} {input_text}"

        final_input = self.tokenizer([final_input_text], return_tensors="pt")
        final_input = {k: v.cuda() for k, v in final_input.items()}

        with torch.inference_mode():
            outputs = self.model.generate(**final_input, **kwargs)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs[0]

if __name__ == "__main__":
    # Example usage
    # dp = DipperParaphraser(model="/work/kalpeshkrish_umass_edu/better-paraphrases/para-paraphrase-ctx-t5-xxl")
    dp = DipperParaphraser()

    prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote valley."
    input_text = "They have never been known to mingle with humans. Today, it is believed these unicorns live in an unspoilt environment which is surrounded by mountains. Its edge is protected by a thick wattle of wattle trees, giving it a majestic appearance. Along with their so-called miracle of multicolored coat, their golden coloured feather makes them look like mirages. Some of them are rumored to be capable of speaking a large amount of different languages. They feed on elk and goats as they were selected from those animals that possess a fierceness to them, and can \"eat\" them with their long horns."

    input_text = prompt.strip() + " <sent> " + input_text.strip() + " </sent>"

    print(f"Input = {input_text}")
    output_l60_greedy = dp.paraphrase(input_text, lex_diversity=60, order_diversity=0, do_sample=False)
    output_l60_o60_greedy = dp.paraphrase(input_text, lex_diversity=60, order_diversity=60, do_sample=False)
    output_l60_sample = dp.paraphrase(input_text, lex_diversity=60, order_diversity=0, do_sample=True, top_p=0.75, top_k=None, max_length=512)
    print(f"Output (Lexical diversity = 60, Greedy) = {output_l60_greedy}\n")
    print(f"Output (Lexical diversity = 60, Order diversity = 60, Greedy) = {output_l60_o60_greedy}\n")
    print(f"Output (Lexical diversity = 60, Sample p = 0.75) = {output_l60_sample}")

    # input_text = "Tracy is a fox. <sent> It is quick and brown. It jumps over the lazy dog. </sent> Tracy also likes to eat cheese."
