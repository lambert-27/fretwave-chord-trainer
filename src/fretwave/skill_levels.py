"""
Purpose: Establishes a dictionary of skill level dictionaries, each containing a chord_accuracy_threshold and transition_target time parameter. 
    A chord gets accepted if the chord's similarity score is above their levels accuracy threshold

Author: Mark Lambert

"""
# a dictionary of dictonaries contianing skill levels and the modifiers for each skill level
skill_levels = {
    "beginner" : {
        "chord_accuracy_threshold": 60,
        "transition_target": None,
    },
    "intermediate" : {
        "chord_accuracy_threshold": 75,
        "transition_target": None,
    },
    "advanced": {
        "chord_accuracy_threshold": 90,
        "transition_target": None,
    },
}

def chord_accepted(level, chord_name, chord_accuracy):
    '''
    Accepts or denies a users chord depending on two conditions: their accuracy score and the level of their playing (beginner | intermediate | advanced) 
        Prints an appropriate message for each chord
    
    :param level: the skill level of the user, as a string (beginner | intermediate | advanced)
    :param chord_name: The name of the detected chord
    :param chord_accuracy: The similarity score returned by the chord detection function
    
    '''
    if chord_accuracy < skill_levels[level]["chord_accuracy_threshold"]:
        print(f"Chord accuracy {chord_accuracy:.2f} is below the threshold for {level} guitar players ({skill_levels[level]["chord_accuracy_threshold"]})")
    else:
        print(f"Chord {chord_name} detected!")