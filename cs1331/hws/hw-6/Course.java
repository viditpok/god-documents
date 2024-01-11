/**
*@author Vidit Pokharna
*@version 1.0
*/
public abstract class Course implements Comparable<Course>, Summarizable {
    /**
     *
     */
    protected String title;

    /**
     *
     */
    protected String subject;

    /**
     *
     */
    protected String courseCode;

    /**
     *
     */
    protected int creditHours;

    /**
     * @param title string
     * @param subject string
     * @param courseCode string
     * @param creditHours int
     */
    public Course(String title, String subject, String courseCode, int creditHours) {
        this.title = title;
        this.subject = subject;
        this.courseCode = courseCode;
        this.creditHours = creditHours;
    }

    /**
     * @return string
     */
    @Override
    public String summarize() {
        String a = String.format("This course is %s. The course is %s credit hours.", title, creditHours);
        return a;
    }

    /**
     * @param course course
     * @return int
     */
    @Override
    public int compareTo(Course course) {
        return courseCode.compareTo(course.courseCode);
    }
}