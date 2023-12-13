//I worked on the assignment alone, using only course-provided materials
/**
*@author Vidit Pokharna
*@version 1.0
*/
public interface Treatable {

    /**
     * @param dogAge int
     * @return int
     */
    static int convertDogToHumanYears(int dogAge) {
        return (int) (16 * Math.log(dogAge) + 31);
    }

    /**
     * @param catAge int
     * @return int
     */
    static int convertCatToHumanYears(int catAge) {
        return (int) (9 * Math.log(catAge) + 18);
    }

    /**
     *
     */
    void treat();
}