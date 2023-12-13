/**
*@author Vidit Pokharna
*@version 1.0
*/
public class ComputerScience extends Course {
    /**
     *
     */
    protected String language;

    /**
     *
     */
    protected boolean hasLab;

    /**
     * @param title string
     * @param subject string
     * @param courseCode string
     * @param creditHours int
     * @param language string
     * @param hasLab boolean
     */
    public ComputerScience(String title, String subject, String courseCode,
    int creditHours, String language, boolean hasLab) {
        super(title, subject, courseCode, creditHours);
        this.language = language;
        this.hasLab = hasLab;
    }

    /**
     * @return string
     */
    public String summarize() {
        String add = "has";
        if (!hasLab) {
            add = "doesn't have";
        }
        String ret = super.summarize();
        ret += String.format(" The language used in this course is %s, and it %s a lab component.", language, add);
        return ret;
    }
}