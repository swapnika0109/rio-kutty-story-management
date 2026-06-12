Need a proper directory struture for this project.
the plan for the project is a agentic workflow using Langchain and Langgrapsh.

Before calling the agents we need helper methods for all LLM which is already implemented in this project they are prompt versioning, circuit breaker, rate limit.

HELPERnTODO:
To update red teaming, gaurdrails, failproof.

Workflow TODO:
workflow1 :  story topics creater agent -> validation agent -> evalution agent -> if evalution fails -> self correction agent.- save the topics
workflow2 :  story creater agent -> validation agent -> evalution agent -> if evalution fails -> self correction agent - save the story in the riostories
Workflow3 : image generator agent -> validation agent ->  evalution agent -> if evalution fails -> self correction agent - save image in the riostories
Workflow4 : audio generator agent -> validation agent ->  evalution agent -> if evalution fails -> self correction agent.- save the audio in the riostories
Workflow5 : activities generator agent -> all existing worflow where we create mcq, art, science, moral. - existing save feature only

Now workflow 1 and 2 are sequential and  is manditory before begining to any other.
once we have workflow 2 result then we can run 3 parellel agents workflow 3,4,5.
if one of the workflow fails on parlell generations we have to retry min 4 times, if still not working we need human in loop.



story model : 
workflow1  - gemini-2.0-flash-lite
workflow2 - gemini-2.5-flash-lite
workflow3 - black-forest-labs/FLUX.1-dev, gemini-2.5-flash-image
Workflow4 - texttospeech api with different voices and different languges.
Workflow5 -    GEMINI_MODEL: str = "gemini-2.0-flash-lite" 
                GEMINI_FALLBACK_MODEL: str = "gemini-2.0-flash"

Lets design this flow as we maintain and track the state manegement and explain why and how we are doing everything.
Leave the prompts to me .
Lets plan


Next TODO:

Is it good to have the api's in main file generally in enterprise level? if not create the necessary package and move the api there.

Move all the prompts in one place so that i can update ad write, if not already done?

TODO:
Explain how can we write prompts for topics and story as we have prompts in two languages.
and inside each langugae we have set of highlevel topics where we take one from the list randomly and genrate an interesting topic on it.


TODO : Starting from story workflow to activities.

Starting point : WF2 - Lets say story is generated and below is the example.
example : 
'{\n  "story": "Come close, Rio. Let me tell you a story that my own grandmother told me, a story older than the oldest trees. It\'s about a time long, long ago, in a land filled with sunshine and singing birds. There was a very kind prince named Rama. He loved his wife, Sita, very much. She was gentle and beautiful, like a flower. One day, a big, scary monster named Ravana, with ten heads and a loud roar, took Sita away. He hid her in a faraway, dark forest. Rama was so sad. His heart felt heavy, like a stone. He heard Sita crying for him. He knew he had to be brave. He looked up at the sky, and saw how the sun always comes back, even after the darkest night. Rio, can you imagine how sad Rama must have felt? He missed Sita so much. He gathered his friends, brave monkeys and strong bears. They travelled through jungles that smelled of sweet flowers and damp earth. They heard the rustle of leaves and the chirping of hidden creatures. Rama\'s eyes were filled with determination. He said, \'I will find Sita and bring her home!\' Ravana was very strong and mean. He thought he could scare Rama away. He sent his monsters to fight Rama. The battle was loud, with clashing sounds and shouts. Rama fought with all his might. He remembered all the good things Sita did. He remembered her sweet smile. This made him even stronger. Rio, can you be strong like Rama when something is not fair?\\n\\nThen, Rama faced Ravana. Ravana was big and frightening. But Rama saw something else. He saw that Ravana’s anger was like a storm, loud and scary, but it would pass. Rama’s love for Sita was like a steady light, always there. He chose to use his light against Ravana’s storm. With a final, brave act, Rama defeated Ravana. The darkness that Ravana brought disappeared. Rama found Sita, safe and sound. She ran to him, and they held each other tight. The whole forest seemed to sigh with relief. The birds sang a happy song. The flowers seemed to bloom brighter. Rama understood that even the biggest, scariest shadows can\'t hide from the light forever. The good inside him, his love and courage, had won. It was like the dawn chasing away the night. Rio, can you see how the light always comes back?",\n  "image_prompt": "Prince Rama with a bow and arrow, standing heroically, facing a ten-headed demon king Ravana in a lush ancient Indian forest, warm golden light filtering through the trees, children\'s book illustration, timeless and gentle",\n  "mcq_seeds": [\n    "Rama fought the bad king Ravana to save his wife Sita.",\n    "Rama was brave and used his love to win.",\n    "The sun always comes back after the night.",\n    "Goodness and love are stronger than anger and meanness."\n  ],\n  "art_seed": "Prince Rama in ancient India forest, holding a bow, facing Ravana, warm golds, earthy tones, side view",\n  "science_concepts": [\n    {\n      "concept": "Goodness as a Guiding Light",\n      "explanation": "Rama\'s goodness was like a light that showed him the way, helping him overcome the darkness of Ravana\'s actions. It is the same way that a lighthouse guides ships safely through the dark sea."\n    },\n    {\n      "concept": "The Nature of Light and Shadow",\n      "explanation": "Light is a form of energy that travels in waves and is always present, even when we cannot see it, like the sun behind clouds. It is the same way that even when you close your eyes, the light is still there."\n    }\n  ],\n  "moral": "Kids, today from this story we have learned that even when bad things seem big and scary, the good and kind things inside us are always stronger, just like Rama\'s love for Sita."\n}'

Observations : Now from this story we know we have impage prompt, lines to use in activities.

Next Steps: 

Story generated -> validate the story (check image, activities) -> evaluate the story (Toxicity, Faithfulness, completeness, engagebility, Misuse, PII Leakage, Role Violation, Bias ) -> self correcttion (if still evaluton failes)-retries, human in loop-> save the story->  start generating image workflow using image prompt  and use story to start generating audio workflow parallely.

Remember this story generation should be in loop and system should generated each story concurrently  and have semaphore atleast 10.

s1 -> image, audio
s2 -> image, audio
s3 -> image, audio
s4 -> mage, audio
s6 -> image, audio
s7 -> image, audio
s8 -> image, audio
s9 -> image, audio
s10 -> image, audio

Next step..
Once the full story including image and audio is genrated saved within the story record.
then start generated activities for that story.
to generated MCQ, use the mcw_seed from the story response and similarly for others( art, scince, moral) instead of story summary or whole story.


Things to check:

if there is a failure in image generation, implement human in loop and also add an individual api to generate image workflow for that story by human. 
Similarly for audio also add an human in loop in the workflow, and also individual api to trigger audio workflow.


the whole story in the databade should look like:

planet_protectors_stories : {
    doc_id : "title"
    story id : "tghcjnskjds...",
    story : ".......",
    image_url : "https://.......",
    audio_url :"https:........"
    theme : 1
    title : "title"
    description : "description",
    topics_id : "..."
}
mindfull_stories : {
    doc_id : "title"
    story id : "tghcjnskjds...",
    story : ".......",
    image_url : "https://.......",
    audio_url :"https:........"
    theme : 1
    title : "title"
    description : "description",
    topics_id : "..."
}
chill_dtories_stories : {
    doc_id : "title"
    story id : "tghcjnskjds...",
    story : ".......",
    image_url : "https://.......",
    audio_url :"https:........"
    theme : 1
    title : "title"
    description : "description"
    topics_id : "..."
}.

Now coming back to topics: 
Update the topics db in this format
planet_protectors_topics : {
    doc_id : "title"
    topics_id: "tghcjnskjds...",
    topics:[{
        title : "",
        description : "",
        theme : "1"
        filter_type : "country",
        filter_value: ["Any"]
    }]
}
mindfull_topics : {
    doc_id : "title"
    topics_id: "tghcjnskjds...",
    topics:[{
        title : "",
        description : "",
        theme : "1"
        filter_type : "country",
        filter_value: ["Any"]
    }]
}
chill_dtories_topics : {
    doc_id : "title"
    topics_id: "tghcjnskjds...",
    topics:[{
        title : "",
        description : "",
        theme : "1"
        filter_type : "country",
        filter_value: ["Any"]
    }]
}

->
Always remove the rio_titles_theme1,2,3 for caching if we need you can use but delete later as we have the above final version.
->
For workflow_checkpoints -> Always clean the checkpoints once the workflow starting from topic to activites  are suceessfull. only keep it to track states once the job is done we dont need it right ? so remove it.

TODO:
Include or implement opensource tracebility like langfuse if it is free or anything else which is free.
Show the tracebility or which step is executing the whole flow.

Audio Generation Steps :
There 2 different type of voices : chirp, standard.
if the request json has chirp as a voice then choose : ["-Chirp3-HD-Gacrux",
		"-Chirp3-HD-Callirrhoe",
		"-Chirp3-HD-Despina",
		"-Chirp3-HD-Iapetus",
		"-Chirp3-HD-Leda",
		"-Chirp3-HD-Zephyr",
		"-Chirp3-HD-Schedar",
		"-Chirp3-HD-Sadaltager",
		"-Chirp3-HD-Rasalgethi",
		"-Chirp3-HD-Umbriel",
		"-Chirp3-HD-Pulcherrima",
		"-Chirp3-HD-Charon",
		"-Chirp3-HD-Zubenelgenubi",
		"-Chirp3-HD-Achird",
		"-Chirp3-HD-Algenib",
		"-Chirp3-HD-Algieba",
		"-Chirp3-HD-Erinome"] 
    else :
    ["-Standard-A",
		"-Standard-B",
		"-Standard-C",
		"-Standard-D"]

        Always choose these randonmly for each story. so the voice should be different and should not match with before or after story.

        The fuk voice should be : language (English :en, Telugu: te)+ "-"+country + voice from above list. 
        eg : en-US-Standard-A

TODO:
make sure for audio for every story we have different voices. when i click next on ui i should actuallys ee differnt voice so double check that


TODO:
Lets say i have topics already generated.
and  generted stories and saved, before running images, audio i have turned the system off. then when i run the app again the stories are again started geenrting, it should first check whther the stories are already generated for those topics are not.
if yes, check image and audio url's if image is missing then only generate image if audio is missing then only generate auido if both are missing then generate both in parellel like today.
thenalso check all activties are generated or nto if not only generate those activties.

One more quesiton:
Lets say i have generated a set of stories are done.
Next time use has entered the same data we show the existing from go applciation, else user has different set of request we generate again using the current workflows.

Test: Lets say i ahve generated 5 topics each next time i wanted to increase to 10 so since 5 is already there remaining 5 has to be generated so it should raise the workflows for rest.
check if this scenario is working or not. 

Test2 :
Fir rest 5 what is the probability of egenrated similar stories that are already there in the system. ideally we should genereate completely new, id the probability is closure what is the solution?

TODO:
Now when i run again if i have stories in the system for the same topics of the reqest json i sent, it shuld trigger masterworkflow only dont generate stories again right?

TODO:
I am sending voice as chirp but the selected voice is standard, please look in to the issue when generating auido it should follow the voice catogery from request. i send chirp then you should use chirp voices only if i send standard you should use the same

TODO:
For audio generation. we need to split the story by paragraphs. and generate audio for each paragraph.
for each paragraph once the audio is generated we have to record start and end timepoints in the belwo structure
audio_timepoints:[{
    "Duration": 4.5828 
"EndTimestamp": 4.5828
"ParagraphNumber": 1
"StartTimestamp": 0
},
{
"Duration": 11.5368
"EndTimestamp": 16.1196
"ParagraphNumber": 2
"StartTimestamp": 4.5828
}
]
combine the audio at the end to further validate and save in a single wav file which we will point out to story record under audio_url as existing.
Also add audio_timepoints to the exsiting story record.


AT each db call check : 
Add exponential backoff + retry in _run_one before giving up — handles transient 429s
Sequential/batched evaluation — fewer concurrent calls


In the request if i ask for theme 1 just generate topics for theme 1 leave rest.

TODO:
Lets update the topics data strcuture inside firestore.
Each theme should have dedicated collection - 
Planet protector topics, Mindfulness topics, Chill topics.
inside each of these collections lets add data in following way : 
insert document by document uuid as an id.
each doucment must have. : {
    age,
    created_at,
    filter_type : 
    filter_value:
    language : English or Telugu
    topics : [
        {
           title : ,
           description :  
           topic_id
        }
    ]
}

Now for stories collection.
Lets create Planet protector stories, Mindfulness stories, Chill stories.
inside each collection lets have document in following way : 
doc uuid as id : 
{
    audio_timepoints : like existing,
    audio_url,
    image_url,
    image_prompt,
    language : English or Telugu,
    mcq_seeds,
    science concepts,
    story_text,
    title,
    topic_id,
    topic document id,
    updated at
    voice
    
}

Currently when generating audio and image there is new collection which is started creating in firestore story_images_v1,story_audio_v1 - these two collections we dont need . Dont create it.

Now coming to workflow checkpoints.
Make sure the checkpoints are created and removed once the step is done.
Lets say if it in validation step o fwf1 then store until validation step once the wf is done remove the checkpoints related to that workflow.

Add checkpoints for all workflows and remove each checkpoints once the work flow is done.

If i run the topics and stoped the server in middle, next time when i start again it should first check the pending workflow from checkpoint and resume from there, instead of checking all the topics generated from the database.

Currently i believe we check the latest data avaioable in the db like topics and if each topic doesnt have story then we create stories. so checking this is really not manageble when dealing with so many data.

So lets completely designed and depend on checkpoints first, if checkpoints are not then start fresh and then always check the generated topc, or story is already there or not.

TODO:

Lets add a prompt for story image generation.
where the story return the image prompt, it shoudl follow exact instruction but the image should attarct age of kid.







