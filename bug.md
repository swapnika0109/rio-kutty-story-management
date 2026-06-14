# BUG-001: activtiies Evalution is failing 

## Status: open
## Priority: high
## Component: activtiies

## Problem
Activities- evalution fix
Analyze the current cost for evalution for each activties.
In real time there might have 1000's of stories and each story might need a lot of evalutions.
SO the main evalution for each type of activty is :
art : the art seed is  aligned with story or not we cna use python -answere relevance
age approriate,
mcq - the art seed is  aligned with story or not we cna use python -answere relevance
age approriate,
moral - is moral conveyed with high engagebility or not, age approriate,
science - is science conveyed with high engagebility or not ,age approriate,

instead of evaluting each activtiy at once. we can map all the activties seed into on paragraph and make one time eval for each activity. i think this way we can reduce cost and evaluate properly.

Reccomend me f this is not right approch.


# BUG-002: Topics and stories in telugu

## Status: open
## Priority: high
## Component: topics and stories in telugu

## Problem2 :
There has been a lot of changes in english prompts that took place recently.
Now lets make all those changes in telugu prompts as well and make sure both prompts are in sync.

## Problem3 :
Currently when generating new topics, it is reusing similar charecters that already used in eixting. The topics should be unique so Let LLM know that she has to create new and unique topics so that we get very unique stories.

If needed can create a firestore method which will give all the titles for that theme and send them to LLM to geenrate new and unique topics.

My question : is this needed at story creation also as do we have to send exiting or using rag a similar stories to Story creator to get not to use similar stories and create unique ones?

## Problem4:

Lets change the topics logic.
Treat this as new enhancement. currently the topics are limited to certain highlevel topics.

There are 3 themes in system.
Planet Protectors : the main idea is to make kids to provide more education and responsibilty on how to treat the planet with love by geerating interesting stories in terms of different natural liviing being, elements, materials, wheathers etc.
So to create new and unique stories, the first step is to create topics which has title and description.
Mindfullness : The main idea is to teach them the way of living by generating stories on  our mythologies, scriptures, vedas or other motivational books. This is knowledge rooting theme where we generate real time stories listed in our history which can actually help us to live better from childhood and also understand the meaning of the precedure like in hindu breaking a coconut what does actually mean those days and why should we still do etc we are doing in daily basis.
Chillstories : The main idea is how to live slow, which actually gives kids to be more creative. these are more interesting stories about daily life style, how actually we should live by geenrating very exciting adventurous stories.

All these theme should actually connect with modern science that no one can really understand and no school actually teach them. for every story we are going to geenrte not only moral but also how it applciable to out daily life and how it is proven scitifically. based on the story we can explain them the science behind it moral behind it.

So the current way is suffcient or not. lets think abou it and lets be more creative to build this
