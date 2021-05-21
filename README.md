# Oxford Flowers 102

The project is based on Udacity's Rubric and used on CLI (Command Line Interface)
by the user to classify variety of flower from an image.

## Install

- Use following commands to install the requirements for the project:

```bash
python -m pip install -r requirements.txt
```

- Make sure you are running with **Python 3.7+**

## Usage

To use the program, use following command:

```bash
python predict.py <image-path> <model-path>
```

This program will represent top 5 features by default. To choose the top
K predictions of class with confidence, use following optional parameter:

```bash
python predict.py image.jpeg model.h5 --top_k 5
```

To use provide class labels for the predictions use following command:

```bash
python predict.py image.jpeg model.h5 --category_names map.json
```

## Credits

@Tejas Kanji ([LordTejas](https://github.com/LordTejas))

## LICENCE

MIT License

Copyright (c) 2018 Udacity

Check the **LICENCE** file to know more.
