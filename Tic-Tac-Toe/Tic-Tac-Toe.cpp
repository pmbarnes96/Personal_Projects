// This program is for playing Tic-Tac-Toe.  It allows human vs human, computer vs computer, or human vs computer play.
// Computer vs computer may be boring because Tic-Tac-Toe, played perfectly, is always a draw, and this program should play perfectly.

#include <cstdlib>
#include <iostream>
#include <vector>

// Dimension of the board
const int N = 3;

// Constants related to user choices and input
const int HUMAN_CHOICE = 0;
const int COMPUTER_CHOICE = 1;
const int MIN_NUM_GAMES = 1;
const int MAX_NUM_GAMES = 10;
const int MIN_MOVE_NUM = 1;
const int MAX_MOVE_NUM = 9;
const int NUM_CHARS_TO_DISCARD = 10000000;

// X (the starting player) shall be represented with a 1, and O shall be presented with a -1.
const int X_PLAYER = 1;
const int O_PLAYER = -1;

// Useful constant for minimax algorithm, just has to be at least as large as the maximum possible gain for either agent
const int MINIMAX_INFINITY = 2;

// The following class defines a game of Tic-Tac-Toe, with the position of the game board and player whose turn it is as attributes,
// and member functions for making a move, determining the winner, and more.
class Game
{
    private:
        int player;
        int board[N][N];
    
    public:
        // Update the game by making a move at row = move.at(0) and column = move.at(1), and switching whose turn it is.
        void update(std::vector<int> move)
        {
            board[move.at(0)][move.at(1)] = player;
            player = -player;
            return;
        }
        
        // Reverse the effect of the update function.
        void downdate(std::vector<int> move)
        {
            board[move.at(0)][move.at(1)] = 0;
            player = -player;
            return;
        }
        
        // Check if a move is valid.
        bool moveIsValid(std::vector<int> move)
        {
            return board[move.at(0)][move.at(1)] == 0;
        }
        
        // Return a list of valid moves given the current board position.
        std::vector<std::vector<int>> returnValidMoves()
        {
            std::vector<std::vector<int>> validMoves = {};
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    if(board[i][j] == 0)
                        {
                            validMoves.push_back({i, j});
                        }
                }
            }
            return validMoves;
        }
        
        // Return 1 if X wins, -1 if O wins, or 0 if there is no winner in the current board position.
        int returnWinner()
        {
            for (int i = 0; i < N; i++)
            {
                if (board[i][0]==X_PLAYER and board[i][1]==X_PLAYER and board[i][2]==X_PLAYER)
                {
                    return X_PLAYER;
                }
            }
            
            for (int j = 0; j < N; j++)
            {
                if (board[0][j]==X_PLAYER and board[1][j]==X_PLAYER and board[2][j]==X_PLAYER)
                {
                    return X_PLAYER;
                }
            }
            
            if (board[0][0]==X_PLAYER and board[1][1]==X_PLAYER and board[2][2]==X_PLAYER)
            {
                return X_PLAYER;
            }
            
            if (board[0][2]==X_PLAYER and board[1][1]==X_PLAYER and board[2][0]==X_PLAYER)
            {
                return X_PLAYER;
            }
            
            for (int i = 0; i < N; i++)
            {
                if (board[i][0]==O_PLAYER and board[i][1]==O_PLAYER and board[i][2]==O_PLAYER)
                {
                    return O_PLAYER;
                }
            }
            
            for (int j = 0; j < N; j++)
            {
                if (board[0][j]==O_PLAYER and board[1][j]==O_PLAYER and board[2][j]==O_PLAYER)
                {
                    return O_PLAYER;
                }
            }
            
            if (board[0][0]==O_PLAYER and board[1][1]==O_PLAYER and board[2][2]==O_PLAYER)
            {
                return O_PLAYER;
            }
            
            if (board[0][2]==O_PLAYER and board[1][1]==O_PLAYER and board[2][0]==O_PLAYER)
            {
                return O_PLAYER;
            }
            
            return 0;
        }
        
        // Return the winner given optimal play from both sides starting from the current board position.
        int minimax()
        {
            std::vector<std::vector<int>> validMoves = returnValidMoves();
            if (validMoves.size() == 0 or returnWinner() != 0)
            {
                return returnWinner();
            }
            
            int value;
            if (player == X_PLAYER)
            {
                value = -MINIMAX_INFINITY;
                for (int i = 0; i < validMoves.size(); i++)
                {
                    update(validMoves.at(i));
                    value = std::max(value, minimax());
                    downdate(validMoves.at(i));
                }
                return value;
            }
            
            else
            {
                value = MINIMAX_INFINITY;
                for (int i = 0; i < validMoves.size(); i++)
                {
                    update(validMoves.at(i));
                    value = std::min(value, minimax());
                    downdate(validMoves.at(i));
                }
                return value;
            }
        }
        
        // From the set of optimal moves given the current position, return a random one, or close enough to random.
        std::vector<int> computeOptimalMove()
        {
            std::vector<int> moveEvaluations = {};
            std::vector<std::vector<int>> validMoves = returnValidMoves();
            int value;
            int maxValue = -MINIMAX_INFINITY;
            std::vector<std::vector<int>> optimalMoves = {};

            for (int i = 0; i < validMoves.size(); i++)
            {
                update(validMoves.at(i));
                value = -player * minimax();
                maxValue = std::max(value, maxValue);
                moveEvaluations.push_back(value);
                downdate(validMoves.at(i));
            }
            
            for (int i = 0; i < moveEvaluations.size(); i++)
            {
                if (moveEvaluations.at(i) == maxValue)
                {
                    optimalMoves.push_back(validMoves.at(i));
                }
            }
            
            return optimalMoves.at(std::rand() % optimalMoves.size());
        }
        
        // Print out the board position in a way most humans would recognize.
        void printBoard()
        {
            for (int i = 0; i < N; i++)
            {
                std::cout << std::endl;
                for (int j = 0; j < N; j++)
                {
                    if (board[i][j] == X_PLAYER)
                    {
                        std::cout << "X ";
                    }
                    
                    else if (board[i][j] == O_PLAYER)
                    {
                        std::cout << "O ";
                    }
                    
                    else
                    {
                        std::cout << "_ ";
                    }
                }
            }
            std::cout << std::endl;
            return;
        }
        
        // Print out whose turn it is.
        void printPlayer()
        {
            if (player == X_PLAYER)
            {
                std::cout << "It is X's turn." << std::endl;
            }
            
            else
            {
                std::cout << "It is O's turn." << std::endl;
            }
        }
        
        // Check if the game is over.
        bool gameOver()
        {
            if (returnWinner() != 0)
                {
                    return true;
                }
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    if (board[i][j] == 0)
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        
        // Return the game to its initial state, with no X's or O's on the board, and X having the next move.
        void clearGame()
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    board[i][j] = 0;
                }
            }
            
            player = X_PLAYER;
        }
        
        // Return 1 if it is X's turn, -1 if it is O's turn.
        int returnPlayer()
        {
            return player;
        }
        
        // Constructor that will always create a game with a blank board, and X having the next move.
        Game()
        {
            clearGame();
        }
};

// Set the input to an integer of the user's choosing in the range [minValue, maxValue].
// Keep asking the user for valid input if invalid input is given.
void takeCinInput(int &input, int minValue, int maxValue)
{
    bool validInputFound = false;

    while (not validInputFound)
    {
        std::cin >> input;
        if (std::cin.fail() or input < minValue or input > maxValue)
        {
            std::cin.clear();
            std::cin.ignore(NUM_CHARS_TO_DISCARD, '\n');
            std::cout << std::endl << "This is not a valid choice." << std::endl;
            std::cout << "Please try again: ";
        }

        else
        {
            validInputFound = true;
        }
    }
}

// Set the input to an integer of the user's choosing in the range [minValue, maxValue].
// Keep asking the user for valid input if invalid input is given.
// This is specifically for making moves in a game, so it checks to see if a move is valid.
void takeCinInput(int &input, int minValue, int maxValue, Game theGame)
{
    bool validInputFound = false;

    while (not validInputFound)
    {
        std::cin >> input;
        if (std::cin.fail() or input < minValue or input > maxValue or (not theGame.moveIsValid({(input-1)/N, (input-1)%N})))
        {
            std::cin.clear();
            std::cin.ignore(NUM_CHARS_TO_DISCARD, '\n');
            std::cout << std::endl << "This is not a valid choice." << std::endl;
            std::cout << "Please try again: ";
        }

        else
        {
            validInputFound = true;
        }
    }
}

int main()
{
    // Declare useful variables for gameplay.
    Game theGame;
    std::vector<int> move;
    int xHumanOrComputer;
    int oHumanOrComputer;
    int numGames;
    int humanMove;
    int winner;
    
    // Allow the user to input settings.
    std::cout << std::endl << "Welcome to Tic-Tac-Toe." << std::endl << std::endl;
    std::cout << "Would you like X, the player that goes first, to be human or computer controlled?" << std::endl;
    std::cout << "Press '0' for human or '1' for computer, and then press 'Enter': ";
    takeCinInput(xHumanOrComputer, HUMAN_CHOICE, COMPUTER_CHOICE);
    std::cout << std::endl << "Would you like O, the player that goes second, to be human or computer controlled?" << std::endl;
    std::cout << "Press '0' for human or '1' for computer, and then press 'Enter': ";
    takeCinInput(oHumanOrComputer, HUMAN_CHOICE, COMPUTER_CHOICE);
    std::cout << std::endl << "How many games would you like to be played?" << std::endl;
    std::cout << "Type in any whole number from 1 to 10, and then press 'Enter': ";
    takeCinInput(numGames, MIN_NUM_GAMES, MAX_NUM_GAMES);
    std::cout << std::endl;
    
    // Instruct a human player how to enter a move.
    if (xHumanOrComputer == HUMAN_CHOICE or oHumanOrComputer == HUMAN_CHOICE)
    {
        std::cout << "When you make a move, you will enter a whole number from 1 to 9." << std::endl;
        std::cout << "This will make a move at the following positions:" << std::endl << std::endl;
        std::cout << "1 2 3" << std::endl;
        std::cout << "4 5 6" << std::endl;
        std::cout << "7 8 9" << std::endl << std::endl;;
    }
    
    // Play however many games the user wanted.
    for (int i = 1; i <= numGames; i++)
    {
        // Start a game.
        theGame.clearGame();
        std::cout << "Game " << i << ":" << std::endl;
        
        // Keep playing the game until it is over.
        while (not theGame.gameOver())
        {
            // Display the board position and whose turn it is.
            std::cout << "The board position is:" << std::endl;
            theGame.printBoard();
            std::cout << std::endl;
            theGame.printPlayer();

            // If it is a human player's turn, have him or her input a move.
            if ((theGame.returnPlayer()==X_PLAYER and xHumanOrComputer==HUMAN_CHOICE) or (theGame.returnPlayer()==O_PLAYER and oHumanOrComputer==HUMAN_CHOICE))
            {
                std::cout << "Enter a move: ";
                takeCinInput(humanMove, MIN_MOVE_NUM, MAX_MOVE_NUM, theGame);
                move = {(humanMove-1)/N, (humanMove-1)%N};
            }
            
            // If it is the computer's turn, have it calculate an optimal move.
            else
            {
                move = theGame.computeOptimalMove();
            }
            
            // Actually make the move, whether it came from a human or computer player.
            theGame.update(move);
        }
        
        // After a game is over, display the final board position, as well as who wins or if it is a draw.
        winner = theGame.returnWinner();
        std::cout << "The final board position is:" << std::endl;
        theGame.printBoard();
        std::cout << std::endl;
        if (winner == X_PLAYER)
        {
            std::cout << "X wins." << std::endl;
        }
        else if (winner == O_PLAYER)
        {
            std::cout << "O wins." << std::endl;
        }
        else
        {
            std::cout << "The result is a draw." << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Let the user choose when to close the program.
    std::cout << "Press the key for any number or letter, then 'Enter' to close the program.";
    std::cin >> humanMove;
}