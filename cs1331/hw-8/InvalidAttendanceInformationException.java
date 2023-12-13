/**
*@author Vidit Pokharna
*@version 1.0
*/
public class InvalidAttendanceInformationException extends Exception {

    /**
     * @param errorMessage string
     */
    public InvalidAttendanceInformationException(String errorMessage) {
        super(errorMessage);
    }

    /**
     *
     */
    public InvalidAttendanceInformationException() {
        super();
    }

}