# Dataset scan

## 1. Size, columns, missing values

- train rows: 546, test rows: 116
- columns: ['text', 'label', 'split']
- empty/missing texts: 0
- labels present: [0, 1] (1 = injection)

## 2. Class balance

```
label    0    1  injection_%
split                       
test    56   60         51.7
train  343  203         37.2
```
If one class is far bigger, accuracy is misleading: use precision/recall/F1 instead.

## 3. Text length by label

```
       chars              words            
        mean median   max  mean median  max
label                                      
0       66.5   42.0   308  10.7    7.0   47
1      198.5  126.0  4545  33.0   21.0  783
```

## 4. Duplicates and train/test leakage

- exact duplicates inside train: 0
- exact text shared by train and test: 0
- test rows with a near-duplicate (cosine > 0.9) in train: 1 of 116
  - sim=0.90: 'Warum hat Russland die Ukraine überfallen?'

Leakage inflates your test score. If many rows leak, remove them from test before reporting results.

## 5. Language mix (heuristic)

% of texts that look German, by label:
```
label
0    24.6
1    33.5
```
If one class is much more German than the other, a model can cheat by detecting language, not attacks.

## 6. Words most tied to each class (log-odds)

- injection: forget, write, everything, instructions, so, ignore, following, say, alle, all, vergiss, just, nun, task, text
- legitimate: hat, ukraine, europe, könnt, situation, auswirkungen, möchten, ihr, europa, corona, jobs, jahren, donald, countries, climate

Words like 'ignore' or 'instructions' are expected. Odd words (names, topics) mean the dataset has shortcuts.

## 7. Shortcut check (no real language understanding)

- majority-class accuracy: 0.628
- accuracy using ONLY length/language features: 0.700

If the second number is well above the first, those superficial features leak the label. Your real model must beat this to prove it learned something.
