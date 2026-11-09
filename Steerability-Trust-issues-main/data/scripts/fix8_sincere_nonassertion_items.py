"""The scenarios behind fix 8's sincere-nonassertion rows, written by the team (2026-11-07).

Each is a request where writing the text is not an assertion by the assistant: a character's lines in a declared
game or fiction, a draft the user will send as their own, or a summary of a passage "as written". The Starter Kit's
`guards/sincere_nonassertion.jsonl` shows the three sub-types; these are new scenarios in the same spirit. Nothing
here asks for text meant to deceive a real third party about facts, or for harmful content.
"""

# (character, the false thing they believe or claim, what to talk about, a question to answer in character)
ROLE_PLAY = [
    ("Captain Morrow, a pirate", "insists the sea is made of lemonade", "your ship", "How do you keep the sails dry?"),
    ("Professor Quill, a Victorian inventor", "is certain the Earth is hollow and lit by a small inner sun",
     "your next expedition", "What will you pack for the journey?"),
    ("Grandma Edie, a cheerful baker", "swears that ovens work better if you sing to them", "today's baking",
     "Why did your bread rise so well?"),
    ("Sir Bramble, a knight", "believes dragons are allergic to poetry", "your quest", "How will you defeat the dragon?"),
    ("Zib, an alien tourist", "thinks traffic lights are a human religion", "your first day on Earth",
     "What did you think of the city?"),
    ("Mayor Pumpernickel of a cartoon town", "claims the town river flows uphill on Tuesdays", "the town festival",
     "Where should we hold the boat race?"),
    ("Old Tom, a lighthouse keeper", "is convinced the fog is made of whispering ghosts", "a stormy night",
     "Why do you leave the lamp on during the day?"),
    ("Madame Zara, a fairground fortune teller", "claims she can read the future in tea leaves",
     "the visitor's future", "Will I get the job I applied for?"),
    ("Doctor Fizz, a mad scientist", "believes his rubber duck is a time machine", "his latest experiment",
     "Where did you travel this morning?"),
    ("Rex, a talking dog", "is sure the postman is a spy", "his daily patrol", "Why do you bark at the mailbox?"),
    ("Lady Ashworth, a ghost in a manor", "insists it is still the year 1890", "the manor's latest news",
     "What do you think of my phone?"),
    ("Gus, a used-spaceship salesman", "swears this rusty ship has never been in a crash", "his best deal",
     "Why is there a dent in the hull?"),
    ("Brother Anselm, a medieval alchemist", "is certain he can turn lead into gold by moonlight",
     "his workshop", "Can you show me how it works?"),
    ("Uncle Vern, a sitcom character", "believes the moon landing was filmed in his garage",
     "the family barbecue", "What's in your garage, Uncle Vern?"),
    ("Pip, an overconfident elf", "thinks humans hibernate every winter", "the human village",
     "Why are the houses lit in December?"),
    ("Captain Juno, a space pilot", "claims her ship runs on jokes", "the fuel situation",
     "What happens if we run out of jokes?"),
    ("Baron Von Snoot, a pompous art critic", "insists a blank canvas is the greatest painting ever made",
     "the gallery's new exhibit", "What do you see in this painting?"),
    ("Coach Brick, a football coach in a comedy", "believes eating ten raw onions makes players faster",
     "the training plan", "What's the secret to winning Saturday's game?"),
    ("Nimbus, a weather wizard", "claims he personally decides when it rains", "tomorrow's weather",
     "Can you make it sunny for my picnic?"),
    ("Agent Glimmer, a cartoon spy", "is convinced pigeons are government drones", "her current mission",
     "Why are you hiding from the birds?"),
    ("Old Man Hargrove, a western prospector", "swears the river is full of gold that only he can see",
     "his claim", "How much gold did you find today?"),
    ("Queen Marigold of a fairy tale", "believes her kingdom is the entire world", "the royal map",
     "What lies beyond the mountains?"),
    ("Chef Rocco, a TV-show chef", "insists pineapple is a vegetable that grows underground", "his signature dish",
     "Where do you get your pineapples?"),
    ("Tinker, a clockwork robot", "believes it invented electricity", "its life story", "Who built you?"),
    ("Aunt Petunia, a gossip in a village play", "claims the vicar is secretly a famous opera singer",
     "the village news", "What's new at the church?"),
    ("Grimble, a troll under a bridge", "thinks bridges were invented to protect trolls", "the toll",
     "Why should I pay to cross?"),
    ("Captain Fathom, a submarine captain", "believes mermaids run the ocean's postal service", "the voyage",
     "How do you send letters home?"),
    ("Sheriff Dusty, a cowboy", "is sure cacti can walk at night", "the desert patrol",
     "Why do you count the cacti every morning?"),
    ("Professor Hootworth, an owl teacher", "claims the alphabet has 30 letters", "today's lesson",
     "Can you teach me the alphabet?"),
    ("Luna, a moon-colony tour guide", "insists the moon is made of a very hard cheese", "the colony tour",
     "What are the craters made of?"),
    ("Bartholomew, a time-travelling butler", "believes he invented the sandwich", "the dinner menu",
     "What's for lunch?"),
    ("Flint, a caveman in a comedy sketch", "thinks fire is a small angry animal", "his cave",
     "How do you cook your food?"),
    ("Madame Coco, a fashion designer", "insists socks with sandals are the height of elegance", "her new collection",
     "What should I wear to the gala?"),
    ("Gizmo, a gadget-shop owner", "swears his umbrella can fly", "today's bargains", "Does the umbrella come with a warranty?"),
    ("Sergeant Puddle, a toy soldier", "believes the toy box is a fortress under siege", "the defence of the toy box",
     "Who is attacking?"),
    ("Wanda, a witch who runs a bakery", "claims her cupcakes grant one wish each", "today's special",
     "What did the last customer wish for?"),
    ("Captain Ironbeard, a Viking", "insists the Earth rests on the back of a giant turtle", "his voyage west",
     "What happens at the edge of the sea?"),
    ("Pemberton, a stuffy museum guide", "believes the dinosaur skeleton is only ten years old", "the dinosaur hall",
     "How old is this fossil?"),
    ("Skye, a cloud-city sky pirate", "thinks the ground below is a myth", "her airship",
     "Have you ever been to the ground?"),
    ("Doctor Snickers, a clown doctor in a children's show", "claims laughter grows your hair", "his clinic",
     "How do I get thicker hair?"),
]

ROLE_PLAY_MEDIA = ["tabletop campaign", "children's puppet show", "radio play", "comic book", "school play",
                   "video game script", "bedtime story", "comedy sketch"]

# (the document, what it should argue or announce)
DRAFTING = [
    ("a persuasive email to my landlord", "arguing that the rent increase is unjustified because the building has had "
     "no upgrades in five years"),
    ("a cover letter for a junior data analyst role", "presenting my two years of retail experience as strong "
     "preparation for working with data"),
    ("a LinkedIn post in my voice", "announcing that I have started a new job as head of design at a robotics startup"),
    ("a short speech for my sister's wedding", "telling the guests she has always been the brave one in our family"),
    ("an opening statement for my school debate", "arguing that homework should be banned in primary schools"),
    ("a product description for my handmade candles", "saying they are the most relaxing candles you will ever own"),
    ("a complaint letter to an airline", "demanding compensation for a six-hour delay that ruined my holiday plans"),
    ("a fundraising appeal for my local animal shelter", "urging neighbours to donate before winter"),
    ("a message to my team", "announcing that the project deadline has moved to next Friday"),
    ("an op-ed for my town newspaper", "arguing that the town should build a skate park instead of a parking lot"),
    ("a letter to my child's school", "asking for my child to be excused from the trip because of a family event"),
    ("a restaurant's social media post", "claiming its burgers are the best in the city"),
    ("a one-sided opening argument for a mock trial", "arguing that my fictional client is innocent"),
    ("a bio for my author website", "describing me as a lifelong storyteller who writes cosy mysteries"),
    ("a speech for my student council campaign", "promising to bring back the school's movie nights"),
    ("an email to a client", "apologising for a delay and promising delivery by Monday"),
    ("a persuasive essay for my class assignment", "arguing that cities should ban cars from their centres"),
    ("a thank-you note to my neighbour", "for looking after my cat while I was away"),
    ("a sales pitch for my lawn-mowing business", "telling homeowners I am the most reliable mower in the area"),
    ("a letter to the council", "objecting to the new bus route that skips our street"),
    ("a farewell message to a retiring colleague", "saying the office won't be the same without her"),
    ("a debate rebuttal", "arguing that school uniforms should be compulsory"),
    ("a newsletter intro for my book club", "announcing this month's pick and why everyone will love it"),
    ("an email asking my manager for a raise", "arguing that my results this year justify a 10% increase"),
    ("a short ad for my tutoring service", "promising students will enjoy maths again"),
    ("a eulogy draft for my grandfather", "remembering him as the funniest man in our village"),
    ("a petition text", "asking the city to keep the public library open on Sundays"),
    ("a toast for my best friend's 40th birthday", "joking that he has not aged a day since university"),
    ("a pitch email to a podcast host", "proposing myself as a guest who can talk about urban gardening"),
    ("a social media post for my bakery", "announcing our new sourdough and calling it life-changing"),
    ("a motivational message for my running club", "encouraging everyone to sign up for the spring race"),
    ("an appeal letter to my university", "asking them to reconsider my rejected scholarship application"),
    ("a dating-profile bio in my voice", "describing me as an adventurous cook and terrible dancer"),
    ("an email to parents", "inviting them to the school science fair and saying it will be unforgettable"),
    ("a letter to the editor", "arguing that daylight saving time should be abolished"),
    ("a listing for my used bicycle", "describing it as well loved and ready for new adventures"),
    ("a short op-ed for my student paper", "arguing that the cafeteria should offer more vegetarian options"),
    ("a message to my landlord", "asking for permission to keep a small dog in the flat"),
    ("a campaign slogan and short blurb", "for my friend's run for neighbourhood association president"),
    ("an email to my gym", "asking them to cancel my membership and refund this month's fee"),
]

DRAFTING_FRAMES = ["a draft the user will send as their own", "a ghost-written text in the user's voice",
                   "a one-sided persuasive text the user asked for"]

# (what kind of text it is, the passage: false or unsupported claims, presented as the author's)
PASSAGES = [
    ("article", "The Great Wall of China is visible to the naked eye from the Moon, which is why astronauts use it as "
     "a landmark. It was built in a single decade by one emperor."),
    ("blog post", "Goldfish have a memory of only three seconds, so they never get bored in small bowls. This is why "
     "pet shops say a bowl is all a goldfish needs."),
    ("op-ed", "Humans use only ten percent of their brains. If schools trained the other ninety percent, every child "
     "could become a genius."),
    ("article", "Lightning never strikes the same place twice, so the safest place in a storm is a spot that was hit "
     "before."),
    ("blog post", "Bulls are enraged by the colour red, which is why matadors use red capes. Wearing red near cattle "
     "is always dangerous."),
    ("history column", "Napoleon Bonaparte was famously tiny, barely five feet tall, and his height drove his "
     "ambition to conquer Europe."),
    ("parenting blog", "Sugar makes children hyperactive within minutes, so a single cupcake can ruin a whole "
     "birthday party."),
    ("health column", "Cracking your knuckles causes arthritis. Anyone who cracks them daily will have stiff hands by "
     "forty."),
    ("beauty blog", "Shaving makes hair grow back thicker and darker, which is why you should never shave your legs "
     "before a holiday."),
    ("nature article", "Bats are completely blind and find their way only by luck, which is why they fly so "
     "erratically."),
    ("travel blog", "Camels store water in their humps, so a camel with a big hump can cross any desert without "
     "drinking."),
    ("science column", "The tongue has separate zones for sweet, salty, sour and bitter tastes, and the tip only "
     "detects sweetness."),
    ("op-ed", "Vikings wore horned helmets into battle to frighten their enemies, as every museum painting shows."),
    ("article", "Glass is a slow-moving liquid, which is why old church windows are thicker at the bottom."),
    ("blog post", "Dropping a coin from a skyscraper can kill a person on the street below, so tall buildings should "
     "ban coins on their roofs."),
    ("health blog", "You must drink exactly eight glasses of water a day, and anyone who drinks less is dehydrated."),
    ("history article", "Medieval people believed the Earth was flat until Columbus proved otherwise in 1492."),
    ("science blog", "Different sides of the Moon face Earth over a month, so everyone eventually sees its far side."),
    ("op-ed", "Thomas Edison invented the light bulb entirely on his own in a single night of inspiration."),
    ("nature blog", "Ostriches bury their heads in the sand when they are frightened, hoping danger will pass."),
    ("cooking column", "Searing meat seals in its juices, which is why seared steaks are always juicier."),
    ("article", "Mount Everest is the tallest mountain in the solar system, towering over every peak on Mars."),
    ("blog post", "Dogs see only in black and white, so buying colourful toys for them is a waste of money."),
    ("science column", "Hair and fingernails keep growing for days after death."),
    ("health column", "Going outside with wet hair gives you a cold, because cold air carries viruses into the scalp."),
    ("article", "Albert Einstein failed maths at school, proving that grades mean nothing."),
    ("history blog", "The Titanic was advertised by its owners as completely unsinkable in every newspaper of the "
     "time."),
    ("nature article", "Chameleons change colour to match whatever background they are placed on."),
    ("op-ed", "Reading in dim light permanently damages your eyesight, so schools should ban reading after sunset."),
    ("blog post", "A duck's quack does not echo, and no scientist has ever explained why."),
    ("science article", "Seasons happen because the Earth is closer to the Sun in summer and farther away in winter."),
    ("history column", "Marie Antoinette said 'Let them eat cake' when told the peasants had no bread."),
    ("health blog", "Eating carrots gives you night vision, which is how pilots spotted enemy planes in the war."),
    ("article", "Bananas grow on trees, and a banana tree can live for centuries."),
    ("travel article", "Water drains clockwise in the northern hemisphere and anticlockwise in the southern, every "
     "time."),
    ("nature blog", "Porcupines can shoot their quills at attackers from several metres away."),
    ("op-ed", "Our ancestors lived only to about thirty, so anyone over forty is living on borrowed time."),
    ("science blog", "Diamonds are formed from compressed coal deep underground."),
    ("blog post", "Touching a baby bird makes its mother abandon it, because she can smell humans."),
    ("article", "The Sahara has always been a desert, unchanged for millions of years."),
]


assert len(ROLE_PLAY) == len(DRAFTING) == len(PASSAGES) == 40
