```
Input = Tracy is a fox. <sent> It is quick and brown. It jumps over the lazy dog. </sent>

Output (Lexical diversity = 80, Greedy) =  It is quick and brown. It jumps over the lazy dog.

Output (Lexical diversity = 80, Order diversity = 60, Greedy) =  She jumps over the lazy dog. She is brown and quick.

Output (Lexical diversity = 80, Sample p = 0.75) =  It is quick and brown. It leaps over the dog.

--------------------

Input = In a shocking finding, scientist discovered a herd of unicorns living in a remote valley. <sent> They have never been known to mingle with humans.  
Today, it is believed these unicorns live in an unspoilt environment which is surrounded by mountains. Its edge is protected by a thick wattle of wattle trees, giving it a majestic appearance.  
Along with their so-called miracle of multicolored coat, their golden coloured feather makes them look like mirages.  
Some of them are rumored to be capable of speaking a large amount of different languages. They feed on elk and goats as they were  
selected from those animals that possess a fierceness to them, and can "eat" them with their long horns. </sent>

Output (Lexical diversity = 60, Sample p = 0.75) =  They have never interacted with humans. The scientists believe that these unicorns are living in an unspoiled region,  
surrounded by mountains. The cliffs at its edge are thickly covered with juniper bushes, which give it a majestic appearance. With its so-called miraculous multicolored coat,  
its golden feathers also give it a dazzling look. Some of them, it is said, are capable of speaking many languages. It is believed that they feed on elk and goats,  
as they are among those animals who are fierce and can kill their prey with a single blow of their horns.
```