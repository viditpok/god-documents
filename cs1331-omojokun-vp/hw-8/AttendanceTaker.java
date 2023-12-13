import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.NoSuchElementException;
import java.util.Scanner;

/**
*@author Vidit Pokharna
*@version 1.0
*/
public class AttendanceTaker {

    /**
     *
     */
    private File inputFile;
    /**
     *
     */
    private File outputFile;

    /**
     * @param inputFile file
     * @param outputFile file
     */
    public AttendanceTaker(File inputFile, File outputFile) {
        this.inputFile = inputFile;
        this.outputFile = outputFile;
    }

    /**
     *
     * @param inputFile string
     * @param outputFile string
     */
    public AttendanceTaker(String inputFile, String outputFile) {
        this(new File(inputFile), new File(outputFile));
    }

    /**
     *
     */
    public void takeAttendance() throws FileNotFoundException {
        Scanner scan = new Scanner(inputFile);
        if (inputFile.length() == 0) {
            scan.close();
            throw new BadFileException("The input file was empty");
        }
        String s;
        try {
            s = scan.nextLine();
        } catch (NoSuchElementException e) {
            throw new BadFileException();
        } finally {
            scan.close();
        }
        if (!((s.substring(0, 3).equals("|--")
            && s.substring((s.length() - 3)).equals("--|")) && (s.length() >= 6))) {
            scan.close();
            throw new BadFileException("The file doesn't have correct beginning or end");
        }
        s = s.substring(3, s.length() - 3);
        String[] names = s.split("---");
        scan.close();

        Scanner scan1 = new Scanner(System.in);
        PrintWriter pw = new PrintWriter(outputFile);
        for (int a = 0; a < names.length; a++) {
            try {
                processStudentAttendance(names[a], scan1, pw);
            } catch (InvalidNameFormatException b) {
                System.out.print(String.format(" Skipping %s because of an invalid name information: %s",
                    names[a], b.getMessage()));
            } catch (InvalidAttendanceInformationException c) {
                System.out.print(String.format(" Skipping %s because of an invalid attendance information: %s",
                    names[a], c.getMessage()));
            }

        }
        scan1.close();
        pw.close();
    }

    /**
     * @param name string
     * @param consoleScanner scanner
     * @param printWriter printwriter
     * @throws InvalidNameFormatException exception thrown
     * @throws InvalidAttendanceInformationException exception thrown
     */
    private static void processStudentAttendance(String name, Scanner consoleScanner, PrintWriter printWriter)
        throws InvalidNameFormatException, InvalidAttendanceInformationException {
        if (!(name.equals(name.toUpperCase()))) {
            printWriter.println("-");
            throw new InvalidNameFormatException("The name isn't uppercase only");
        } else {
            for (int a = 0; a < name.length(); a++) {
                if (name.charAt(a) >= 48 && name.charAt(a) <= 57) {
                    printWriter.println("-");
                    throw new InvalidNameFormatException("The name has a digit");
                } else if (name.charAt(a) == 124) {
                    printWriter.println("-");
                    throw new InvalidNameFormatException("The name has a pipe character");
                }
            }
        }

        System.out.print(name + ": ");
        String userEnter = consoleScanner.nextLine();
        if (userEnter.equals("")) {
            printWriter.println("-");
            throw new InvalidAttendanceInformationException("Attendance information missing");
        } else if (!(userEnter.equals("A") || userEnter.equals("P"))) {
            printWriter.println("-");
            throw new InvalidAttendanceInformationException("Attendance information is not P or A");
        } else {
            printWriter.println(userEnter);
        }
    }

    /**
     * @param args string[]
     * @throws FileNotFoundException exception thrown
     */
    public static void main(String[] args) throws FileNotFoundException {
        AttendanceTaker at = new AttendanceTaker(args[0], args[1]);
        at.takeAttendance();
    }
}