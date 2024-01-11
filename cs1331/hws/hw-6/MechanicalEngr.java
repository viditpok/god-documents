/**
*@author Vidit Pokharna
*@version 1.0
*/
public class MechanicalEngr extends Course {
    /**
     *
     */
    protected String[] equations;

    /**
     * @param title string
     * @param subject string
     * @param courseCode string
     * @param creditHours int
     * @param equations string[]
     */
    public MechanicalEngr(String title, String subject, String courseCode, int creditHours, String[] equations) {
        super(title, subject, courseCode, creditHours);
        this.equations = equations;
    }

    /**
     * @return string
     */
    public String summarize() {
        String add = "";
        for (int a = 0; a < equations.length; a++) {
            if (a != equations.length - 1) {
                add += equations[a] + ", ";
            } else {
                add += equations[a] + ".";
            }
        }
        String ret = super.summarize() + " This course uses these equation(s): " + add;
        return ret;
    }
}