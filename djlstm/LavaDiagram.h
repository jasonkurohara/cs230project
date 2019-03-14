/**
 * Edit this file to provide answers as to where each of the indicated functions
 * sits in the Venn diagram provided in the problem set handout. You will likely
 * want to read the comments on the Position type for more information about this.
 *
 * This header file does not contain any local test cases. You'll need to submit
 * it to the autograder on GradeScope to get feedback on it.
 */

#ifndef LavaDiagram_Included
#define LavaDiagram_Included

/**
 * Type: Position
 *
 * An enumerated type representing a position in the lava diagram.
 */
enum class Position {
    Regular,                  // It's a regular language (in REG).
    DecidableNotRegular,      // It's decidable, but not regular (in R, not REG).
    RecognizableNotDecidable, // It's recognizable, but not decidable (in RE, not R)
    Unrecognizable,           // It's unrecognizable (not in RE)
    TODO_AnswerMe             // Default value given when you haven't yet provided an answer.
};

/**
 * Location of (1) in the Venn diagram. It's a great example of a regular language.
 * (Do you see why? And a good question to ponder - this language is infinite, but
 * it's also regular. How is that possible, given what you proved in PS7 about
 * finite languages?)
 *
 * Notice that the answer is provided by writing Position::Regular. This is the syntax
 * you'll use to select which option you think applies.
 */
const Position q1 = Position::Regular;

/**
 * Location of (2) in the Venn diagram. It's not an RE language, since we specifically
 * constructed it to be a non-RE language!
 *
 * Again, notice the use of the syntax Position::Unrecognizable to indicate what the answer is.
 */
const Position q2 = Position::Unrecognizable;

/* You need to edit the answers below. */

const Position q3  = Position::Decidable;
const Position q4  = Position::DecidableNotRegular;
const Position q5  = Position::DecidableNotRegular;
const Position q6  = Position::DecidableNotRegular;
const Position q7  = Position::Regular;
const Position q8  = Position::Unrecognizable;
const Position q9  = Position::RecognizableNotDecidable;
const Position q10 = Position::Unrecognizable;
const Position q11 = Position::Unrecognizable;
const Position q12 = Position::Regular;


#endif