import java.util.Arrays;
/**
*@author Vidit Pokharna
*@version 1.0
*/
public class CourseCatalog {

    /**
     *
     */
    protected Course[] catalog;

    /**
     * @param catalog course[]
     */
    public CourseCatalog(Course[] catalog) {
        Arrays.sort(catalog);
        this.catalog = catalog;
    }

    /**
     *
     */
    public CourseCatalog() {
        Course[] array = {};
        this.catalog = array;
    }

    /**
     *
     */
    public void browseCourseCatalog() {
        for (int a = 0; a < catalog.length; a++) {
            String ret = catalog[a].courseCode + ": " + catalog[a].summarize();
            System.out.println(ret);
        }
    }

    /**
     * @param newCourse course
     */
    public void addCourse(Course newCourse) {
        Course[] catalog1 = new Course[catalog.length + 1];
        if (catalog.length != 0) {
            for (int a = 0; a < catalog.length; a++) {
                catalog1[a] = catalog[a];
            }
            catalog1[catalog.length + 1] = newCourse;
            Arrays.sort(catalog1);
        } else {
            catalog1[0] = newCourse;
        }
        catalog = catalog1;
    }

    /**
     * @param courseCode string
     * @return course
     */
    public Course getCourse(String courseCode) {
        Course b = null;
        for (int a = 0; a < catalog.length; a++) {
            if (catalog[a].courseCode.equals(courseCode)) {
                b = catalog[a];
                break;
            }
        }
        return b;
    }

    /**
     * @return int
     */
    public int getNumberOfCourses() {
        return catalog.length;
    }
}