#!/bin/bash

#MODEL=dolphin-2.9.4-llama3.1-8b-Q2_K.gguf
#MODEL=dolphin-2.9.4-llama3.1-8b-Q3_K_S.gguf
MODEL=dolphin-2.9.4-llama3.1-8b-Q4_K_M.gguf

TITLE1="Title: The Curious Squirrel's Adventure\nIn a cozy forest filled with tall trees and whispering leaves, there lived a curious little squirrel named Sammy. Sammy had always wondered what lay beyond the great oak tree at the edge of the forest, where the sunlight danced in a way he'd never seen before"

TITLE2="Title: Luna the Brave Little Mouse\nDeep in the heart of the meadow, under a blanket of wildflowers, lived a tiny mouse named Luna. Luna wasn't the biggest or the strongest, but she had a heart full of courage. One day, she decided to venture out to find the legendary cheese that was said to glow like the moon"

TITLE3="Title: Oliver the Owl and the Midnight Mystery\nHigh up in the tallest tree of Whispering Woods, Oliver the Owl kept watch over the nighttime world. But one night, something strange happened—Oliver heard a sound he had never heard before, a soft and mysterious music coming from the heart of the forest"

TITLE4="Title: Benny the Bear and the Lost Honey\nBenny the Bear loved nothing more than the sweet taste of honey. But one morning, when he went to check on his favorite honey tree, the honey was gone!"

TITLE5="Title: Fluffy the Fox and the Secret of the River\nFluffy the Fox had always been told never to go near the river, where the water rushed and roared. But one summer day, as Fluffy was playing near the edge, she saw something sparkling in the water. Curiosity tugged at her, and she wondered what secrets the river might hold"

PROMPT="### Write a short fairy tale of about 500 words suitable for little kids.\n$TITLE4"

/usr/bin/time -v ./llama-cli -m $MODEL -p "$PROMPT" -c 800 -n 800 -r "###"

