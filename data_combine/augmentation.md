### Problem Identified
- Some merged captions were simply concatenating original captions with "and" instead of semantically synthesizing them into a unified description.
### Improvements Implemented
1. Model Upgrade
- Upgraded from gpt-4o-mini to gpt-4o for better understanding and generation capabilities
2. Prompt Optimization
- Explicitly prohibits simple concatenation (using "and" or commas)
- Requires identifying common themes and merging overlapping information
- Emphasizes output as a single unified scene description
3. Few-shot Examples
- Added 3 examples demonstrating both bad and good merging approaches
- Covers diverse scenarios (vehicles, animals, landscapes)
- Helps the model understand the expected output format
4. Parameter Adjustment
- Increased max_tokens from 60 to 120 to provide more space for natural expression
