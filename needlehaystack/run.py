from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from jsonargparse import CLI

# from . import LLMNeedleHaystackTester, LLMMultiNeedleHaystackTester
from . import LLMNeedleHaystackTester
# from .evaluators import Evaluator, LangSmithEvaluator, OpenAIEvaluator
from .evaluators import Evaluator, OpenAIEvaluator
from .baselines import ViLLMBaseModel, GPT4V
# from .providers import Anthropic, ModelProvider, OpenAI, Cohere

load_dotenv()

@dataclass
class CommandArgs():
    provider: str = "GPT4V"
    evaluator: str = "openai"
    evaluator_model_name: Optional[str] = "gpt-35-turbo-0125"
    # needle: Optional[str] = "\nThe needle word is 'secret'.\n"
    # needle_modality: Optional[str] = "text"
    # retrieval_question: Optional[str] = "What is the needle word?"
    needle: Optional[str] = "insert.mp4"
    needle_desc: Optional[str] = "yes"
    needle_modality: Optional[str] = "video"
    retrieval_question: Optional[str] = "Does the dog bite the frisbee?"
    haystack_dir: Optional[str] = "Ego"
    results_version: Optional[int] = 1
    context_lengths_min: Optional[int] = 300
    context_lengths_max: Optional[int] = 3000
    context_lengths_num_intervals: Optional[int] = 35
    context_lengths: Optional[list[int]] = None
    video_depth_percent_min: Optional[int] = 0
    video_depth_percent_max: Optional[int] = 100
    video_depth_percent_intervals: Optional[int] = 35
    video_depth_percents: Optional[list[int]] = None
    video_depth_percent_interval_type: Optional[str] = "linear"
    num_concurrent_requests: Optional[int] = 1
    save_results: Optional[bool] = True
    save_contexts: Optional[bool] = True
    final_context_length_buffer: Optional[int] = 200
    seconds_to_sleep_between_completions: Optional[float] = None
    print_ongoing_status: Optional[bool] = True
    # LangSmith parameters
    eval_set: Optional[str] = "multi-needle-eval-pizza-3"
    # Multi-needle parameters
    multi_needle: Optional[bool] = False # TODO: optimize multi_needle in video
    needles: list[str] = field(default_factory=lambda: [
        " Figs are one of the secret ingredients needed to build the perfect pizza. ", 
        " Prosciutto is one of the secret ingredients needed to build the perfect pizza. ", 
        " Goat cheese is one of the secret ingredients needed to build the perfect pizza. "
    ])

def get_model_to_test(args: CommandArgs) -> ViLLMBaseModel:
    """
    Determines and returns the appropriate model provider based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        ModelProvider: An instance of the specified model provider class.
    
    Raises:
        ValueError: If the specified provider is not supported.
    """
    match args.provider.lower():
        case "gpt4v":
            return GPT4V({"model_path": None, "device": 0})
        case _:
            raise ValueError(f"Invalid provider: {args.provider}")

def get_evaluator(args: CommandArgs) -> Evaluator:
    """
    Selects and returns the appropriate evaluator based on the provided command arguments.
    
    Args:
        args (CommandArgs): The command line arguments parsed into a CommandArgs dataclass instance.
        
    Returns:
        Evaluator: An instance of the specified evaluator class.
        
    Raises:
        ValueError: If the specified evaluator is not supported.
    """
    match args.evaluator.lower():
        case "openai":
            if args.needle_desc:
                return OpenAIEvaluator(model_name=args.evaluator_model_name,
                                    question_asked=args.retrieval_question,
                                    true_answer=args.needle_desc)
            else:
                return OpenAIEvaluator(model_name=args.evaluator_model_name,
                                    question_asked=args.retrieval_question,
                                    true_answer=args.needle)
        case "langsmith":
            return LangSmithEvaluator()
        case _:
            raise ValueError(f"Invalid evaluator: {args.evaluator}")

def main():
    """
    The main function to execute the testing process based on command line arguments.
    
    It parses the command line arguments, selects the appropriate model provider and evaluator,
    and initiates the testing process either for single-needle or multi-needle scenarios.
    """
    args = CLI(CommandArgs, as_positional=False)
    args.model_to_test = get_model_to_test(args)
    args.evaluator = get_evaluator(args)
    
    if args.multi_needle == True:
        print("Testing multi-needle")
        tester = LLMMultiNeedleHaystackTester(**args.__dict__)
    else: 
        print("Testing single-needle")
        tester = LLMNeedleHaystackTester(**args.__dict__)
    tester.start_test()

if __name__ == "__main__":
    main()
