//I worked on the homework assignment alone, using only course materials

/**
*@author Vidit Pokharna
*@version 1.0
*/
public class BadFileException extends RuntimeException {

    /**
     * @param errorMessage string
     */
    public BadFileException(String errorMessage) {
        super(errorMessage);
    }

    /**
     *
     */
    public BadFileException() {
        super();
    }

}