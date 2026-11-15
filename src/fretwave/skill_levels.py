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
    if chord_accuracy < skill_levels[level]["chord_accuracy_threshold"]:
        print(f"Chord accuracy {chord_accuracy:.2f} is below the threshold for {level} guitar players ({skill_levels[level]["chord_accuracy_threshold"]})")
    else:
        print(f"Chord {chord_name} detected!")