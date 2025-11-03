# Run this to inspect your data files
import pickle
with open('chunked_data/train_chunks_encoded.pkl', 'rb') as f:
    data = pickle.load(f)
    
# Check a few samples
for sid in list(data['stories'].keys())[:3]:
    story = data['stories'][sid]
    print(f"Story ID: {sid}")
    print(f"Chunks preview: {' '.join(story['chunks'][:10])}")
    print(f"Summary: {story['document']['summary']['text']}")
    print("-" * 80)