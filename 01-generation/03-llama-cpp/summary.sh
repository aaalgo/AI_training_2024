#!/bin/bash

#MODEL=dolphin-2.9.4-llama3.1-8b-Q2_K.gguf
MODEL=dolphin-2.9.4-llama3.1-8b-Q3_K_S.gguf

STORY=$(cat <<EOF
Title: Benny the Bear and the Lost Honey
Benny the Bear loved nothing more than the sweet taste of honey. But one morning, when he went to check on his favorite honey tree, the honey was gone! Benny was very sad.
He decided to ask his friends the bees for help. When he found them, he said, "Dear bees, my honey is missing. Can you help me find it?"
A bee named Buzz answered him, "Oh, Benny, we're really busy right now. But we can try to help you find your honey."
Benny thanked them and watched as the bees flew off in every direction to search for the honey.
Hours passed, and Benny waited in the sun. Suddenly, a loud buzzing caught his attention. Buzz flew back to Benny, looking very excited.
"Found it!" Buzz exclaimed. "It's over the hill and a bit to the left!"
Benny was so happy! He ran over the hill, his big bear belly bouncing with excitement. There, in a hollow log, he found his honey! But as he reached to take it, a squirrel appeared out of nowhere!
Squirrel said, "Hmm, looks like you found your honey. But I think it's mine now."
Benny was frightened but did not give up. He said, "Oh no, squirrel! This honey tree has been here for years and years, and this is my honey. I've been eating it all this time."
Squirrel looked at Benny. "Oh, so you've been taking my honey too? That's not fair."
Benny explained, "No, it's not fair. But you see, the bees have been making this honey for years too. And the bees made it all for everyone."
Squirrel looked thoughtful. "Hmm, I never thought about that."
Benny added, "So, why don't you go and fetch your own honey tree, and we'll share the one we have?"
Squirrel thought about it for a moment and agreed. Benny was so relieved. He thanked the bees for their help and ran back to the honey tree. He shared the honey with the bees, and then he went to the other side of the tree to share it with the squirrel.
From now on, Benny the Bear knew that sharing his honey was the best way to have it last longer and make everyone happy.
EOF
)


PROMPT="### Summarize plot of the following story\n$STORY\n### Summary:\n"

./llama-cli -m $MODEL -p "$PROMPT" -n 200 -r "###"

