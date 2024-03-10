@echo off
echo === Acquiring datasets ===
echo ---

if not exist data mkdir data
cd data

echo - Downloading WikiText-2 (WT2)
if not exist wikitext-2 (
    mkdir wikitext-2
    cd wikitext-2
    curl -LO https://github.com/iliaschalkidis/ELMo-keras/raw/master/data/datasets/wikitext-2/wiki.test.tokens
    curl -LO https://github.com/iliaschalkidis/ELMo-keras/raw/master/data/datasets/wikitext-2/wiki.train.tokens
    curl -LO https://github.com/iliaschalkidis/ELMo-keras/raw/master/data/datasets/wikitext-2/wiki.valid.tokens
    ren wiki.train.tokens train.txt
    ren wiki.valid.tokens valid.txt
    ren wiki.test.tokens test.txt
    cd ..
)

echo - Downloading WikiText-103 (WT103)
if not exist wikitext-103 (
    tar -xzf %HOMEPATH%\Downloads\wikitext-103.tar.gz -C .
    cd wikitext-103
    ren wiki.train.tokens train.txt
    ren wiki.valid.tokens valid.txt
    ren wiki.test.tokens test.txt
    cd ..
)

echo - Downloading enwik8 (Character)
if not exist enwik8 (
    mkdir enwik8
    cd enwik8
    curl -LO http://mattmahoney.net/dc/enwik8.zip
    curl -LO https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
    python prep_enwik8.py
    del /f /q enwik8.zip
    cd ..
)

echo - Downloading text8 (Character)
if not exist text8 (
    mkdir text8
    cd text8
    curl -LO http://mattmahoney.net/dc/text8.zip
    curl -LO https://raw.githubusercontent.com/kimiyoung/transformer-xl/master/prep_text8.py
    python prep_text8.py
    del /f /q text8.zip
    cd ..
)

echo - Downloading Penn Treebank (PTB)
if not exist penn (
    curl -LO http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    tar -xzf simple-examples.tgz -C .
    mkdir penn
    cd penn
    move ..\simple-examples\data\ptb.train.txt train.txt
    move ..\simple-examples\data\ptb.test.txt test.txt
    move ..\simple-examples\data\ptb.valid.txt valid.txt
    cd ..
    rd /s /q simple-examples
    del /f /q simple-examples.tgz
)

@REM echo - Downloading 1B words
@REM if not exist one-billion-words (
@REM     mkdir one-billion-words
@REM     cd one-billion-words
@REM     curl -LO http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz
@REM     tar xzvf 1-billion-word-language-modeling-benchmark-r13output.tar.gz -C .
@REM     set path=1-billion-word-language-modeling-benchmark-r13output\heldout-monolingual.tokenized.shuffled
@REM     type %path%\news.en.heldout-00000-of-00050 > valid.txt
@REM     type %path%\news.en.heldout-00000-of-00050 > test.txt

@REM     curl -LO https://raw.githubusercontent.com/rafaljozefowicz/lm/master/1b_word_vocab.txt
@REM     cd ..
@REM )

echo ---
echo Happy language modeling :)