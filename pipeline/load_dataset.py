from typing import Optional, Union, Dict, Any
import logging


from datasets import load_dataset, Dataset


logger = logging.getLogger(__name__)


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

    # Group by story/document id
    stories: Dict[str, Any] = {}
    for item in ds:
        story_id = item['document']['id']
        if story_id not in stories:
            stories[story_id] = {
                'document': item['document'],
                'questions': [],
                'answers': []
            }
        stories[story_id]['questions'].append(item['question'])
        stories[story_id]['answers'].append(item['answers'])

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
