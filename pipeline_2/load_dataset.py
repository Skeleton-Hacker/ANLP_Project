from typing import Optional, Union, Dict, Any
import logging
import re


from datasets import load_dataset, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm


logger = logging.getLogger(__name__)


def clean_text(s: str) -> str:
    """Clean text by removing BOMs, normalizing whitespace, and handling special characters.
    
    Args:
        s: Input string to clean
        
    Returns:
        Cleaned string with normalized whitespace
    """
    if not isinstance(s, str):
        return s
    
    # Remove Unicode BOM and common mis-decoded BOM sequences
    s = s.replace('\ufeff', '')
    s = s.replace('ï»¿', '')
    
    # Normalize line breaks to spaces and remove carriage returns
    s = s.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    
    # Replace underscores with spaces (user requested handling)
    s = s.replace('_', ' ')
    
    # Collapse multiple whitespace into single space and strip
    s = re.sub(r"\s+", ' ', s).strip()
    
    return s


def _load(
    dataset_name: str,
    split: str,
    max_samples: Optional[int] = None,
    trust_remote_code: bool = False,
    group_by_story: bool = True,
) -> Union[Dataset, Dict[str, Any]]:
    """Internal helper to load a single split and optionally limit samples.

    Args:
        dataset_name: name of the dataset (as passed to `datasets.load_dataset`).
        split: split name ("train", "validation", "test", etc.).
        max_samples: if provided, select the first `max_samples` examples.
        trust_remote_code: forward to `load_dataset` when loading remote code.
        group_by_story: when True, convert the loaded dataset into a dictionary
            keyed by story id with the structure::

                {
                  story_id: {
                    'document': <document dict>,
                    'questions': [q1, q2, ...],
                    'answers': [a1, a2, ...]
                  },
                  ...
                }

            This mirrors the requested reformat where a story's summary is
            available at `stories[story_id]['document']['summary']`.

    Returns:
        Either the original `datasets.Dataset` (if group_by_story is False) or
        the grouped dictionary when True.
    """
    logger.info("Loading dataset %s split=%s", dataset_name, split)

    ds = load_dataset(dataset_name, split=split, trust_remote_code=trust_remote_code)
  

    if max_samples is not None:
        n = min(max_samples, len(ds))
        logger.info("Selecting first %d samples (of %d)", n, len(ds))
        ds = ds.select(range(n))

    if len(ds) > 0:
        try:
            keys = list(ds[0].keys())
        except Exception:
            keys = []
    else:
        keys = []

    logger.info("Loaded dataset with %d examples. Sample keys: %s", len(ds), keys)

    if not group_by_story:
        return ds

    # Group by story/document id (multithreaded)
    logger.info("Restructuring dataset by story ID and cleaning text...")

    stories: Dict[str, Any] = {}
    lock = threading.Lock()

    def _process(item):
        story_id = item['document']['id']
        q = clean_text(item['question'])
        a = clean_text(item['answers'][0]['text']) if item['answers'] and len(item['answers']) > 0 else item['answers']
        doc = item['document'].copy()
        
        # Commenting out since it takes too much time to process the entire dataset
        # if 'text' in doc:
        #     doc['text'] = clean_text(doc['text'])
        # if 'summary' in doc and isinstance(doc['summary'], dict) and 'text' in doc['summary']:
        #     doc['summary']['text'] = clean_text(doc['summary']['text'])
        
        with lock:
            if story_id not in stories:
                stories[story_id] = {
                    'document': doc,
                    'questions': [],
                    'answers': []
                }
            stories[story_id]['questions'].append(q)
            stories[story_id]['answers'].append(a)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_process, item) for item in ds]
        
        # Use tqdm to show progress
        with tqdm(total=len(futures), desc=f"Processing {split} examples", unit="ex") as pbar:
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                pbar.update(1)

    logger.info("Total stories after restructuring: %d", len(stories))
    return stories


def load_train(
    dataset_name: str = "deepmind/narrativeqa",
    max_samples: Optional[int] = None,
    trust_remote_code: bool = True,
    group_by_story: bool = True,
) -> Union[Dataset, Dict[str, Any]]:
    """Load the training split for `dataset_name`.

    Defaults mirror other repo files which use the NarrativeQA dataset.
    """
    return _load(
        dataset_name,
        "train",
        max_samples=max_samples,
        trust_remote_code=trust_remote_code,
        group_by_story=group_by_story,
    )


def load_validation(
    dataset_name: str = "deepmind/narrativeqa",
    max_samples: Optional[int] = None,
    trust_remote_code: bool = True,
    group_by_story: bool = True,
) -> Union[Dataset, Dict[str, Any]]:
    """Load the validation split for `dataset_name`."""
    # Some datasets use "validation", others "validation[:10%]" or "dev"; callers can override dataset_name/split if needed
    return _load(
        dataset_name,
        "validation",
        max_samples=max_samples,
        trust_remote_code=trust_remote_code,
        group_by_story=group_by_story,
    )


def load_test(
    dataset_name: str = "deepmind/narrativeqa",
    max_samples: Optional[int] = None,
    trust_remote_code: bool = True,
    group_by_story: bool = True,
) -> Union[Dataset, Dict[str, Any]]:
    """Load the test split for `dataset_name`."""
    return _load(
        dataset_name,
        "test",
        max_samples=max_samples,
        trust_remote_code=trust_remote_code,
        group_by_story=group_by_story,
    )


__all__ = ["load_train", "load_validation", "load_test"]

if __name__ == "__main__":
    train = load_train()
    print("Length of Train: ",len(train))
    validation = load_validation()
    print("Length of Val: ",len(validation))
    test = load_test()
    print("Length of Test: ",len(test))

    # total number of questions per split
    total_train_questions = sum(len(story['questions']) for story in train.values())
    total_validation_questions = sum(len(story['questions']) for story in validation.values())
    total_test_questions = sum(len(story['questions']) for story in test.values())
    print(f"Total training questions: {total_train_questions}")
    print(f"Total validation questions: {total_validation_questions}")
    print(f"Total test questions: {total_test_questions}")
