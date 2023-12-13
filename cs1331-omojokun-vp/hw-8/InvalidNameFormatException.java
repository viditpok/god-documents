/**
*@author Vidit Pokharna
*@version 1.0
*/
public class InvalidNameFormatException extends Exception {

    /**
     * @param errorMessage string
     */
    public InvalidNameFormatException(String errorMessage) {
        super(errorMessage);
    }

    /**
     *
     */
    public InvalidNameFormatException() {
        super();
    }

}