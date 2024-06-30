import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

class TabularDataGenerator:
    """
    A class to generate synthetic tabular data using a language model.

    Attributes:
        device (torch.device): The device (CPU or GPU) to run the model on.
        use_fp16 (bool): Whether to use FP16 precision.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModelForCausalLM): The language model for data generation.
    """

    def __init__(self, model_name: str, device: str = 'cpu', use_fp16: bool = False):
        """
        Initializes the TabularDataGenerator with the specified model.

        Args:
            model_name (str): The name of the pre-trained model to use.
            device (str): The device to run the model on ('cpu' or 'cuda').
            use_fp16 (bool): Whether to use FP16 precision.
        """
        self.device = torch.device(device)
        self.use_fp16 = use_fp16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        if self.use_fp16:
            self.model = self.model.half()

    def preprocess_prompt(self, prompt: str) -> torch.Tensor:
        """
        Preprocesses the input prompt for the model.

        Args:
            prompt (str): The input prompt string.

        Returns:
            torch.Tensor: The tokenized input tensor.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        return inputs

    def generate_data(self, prompt: str, num_samples: int, max_length: int = 1024) -> List[str]:
        """
        Generates synthetic data based on the prompt.

        Args:
            prompt (str): The input prompt string.
            num_samples (int): The number of samples to generate.
            max_length (int): The maximum length of the generated sequences.

        Returns:
            List[str]: A list of generated data strings.
        """
        inputs = self.preprocess_prompt(prompt)
        generated_data = []
        while len(generated_data) < num_samples:
            outputs = self.model.generate(
                inputs['input_ids'], 
                max_length=min(max_length, 1024), 
                num_return_sequences=1, 
                do_sample=True, 
                temperature=0.7
            )
            generated_lines = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n')
            generated_data.extend(generated_lines)
        return generated_data[:num_samples]

    def postprocess_output(self, generated_data: List[str]) -> pd.DataFrame:
        """
        Converts the generated data into a DataFrame.

        Args:
            generated_data (List[str]): The list of generated data strings.

        Returns:
            pd.DataFrame: The DataFrame containing the generated data.
        """
        rows = [row.split(',') for row in generated_data]
        return pd.DataFrame(rows)

def main():
    """
    The main function to generate synthetic tabular data and save it to a CSV file.
    """
    model_name = "gpt2"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_fp16 = torch.cuda.is_available()
    generator = TabularDataGenerator(model_name=model_name, device=device, use_fp16=use_fp16)

    prompt = """
    Continue generating synthetic applicant information in the following format. Make sure the data is realistic, fair and unbiased.:
        AppID, Ins_Age, Ins_Gender, Ht, Wt, Issue_Date

    Where:
        AppID: Anonymized Applicant ID
        Ins_Age: Applicant Age
        Ins_Gender: Applicant Gender (Male/Female)
        Ht: Height in the format '507' which means 5 feet 7 inches
        Wt: Weight of the Applicant in pounds
        Issue_Date: Date of Application in YYYY-MM-DD format

    Example Data:
        001,25,Male,507,150,2022-01-01
        002,30,Female,502,130,2022-02-01
        003,22,Male,510,180,2022-03-01
        004,28,Female,505,120,2022-04-01
        005,35,Male,600,200,2022-05-01
    """

    num_samples = 100
    generated_data = generator.generate_data(prompt, num_samples)
    df = generator.postprocess_output(generated_data)

    output_path = f'generated_data_{num_samples}.csv'
    df.to_csv(output_path, index=False)
    print(f"Synthetic data generation complete. Data saved to {output_path}")

if __name__ == "__main__":
    main()
