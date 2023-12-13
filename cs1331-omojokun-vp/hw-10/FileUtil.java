import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.List;

/**
 * @author CS1331 Fall 2020 TAs
 * @version 1.0
 * This class contains methods to facilitate saving a List of StartUpIdea's to a file
 */
public class FileUtil {

    /**
     * Traverses through the List of StartUpIdeas and saves each ideas toString()
     * to the specified file
     * @param ideas list of StartUpIdea's
     * @param file to save StartUpIdea's
     * @return true if successful, false if unsuccessful
     */
    public static boolean saveIdeasToFile(List<StartUpIdea> ideas, File file) {
        PrintWriter printWriter = null;
        try {
            printWriter = new PrintWriter(file);
        } catch (FileNotFoundException e) {
            System.out.println(e.toString());
            return false;
        }
        for (int i = 0; i < ideas.size(); i++) {
            printWriter.println((i + 1) + ":");
            printWriter.println(ideas.get(i).toString());
        }
        printWriter.close();
        return true;
    }
}