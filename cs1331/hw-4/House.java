//I worked on the assignment alone, using only course-provided materials
/**
*This class represents the driver class
*@author Vidit Pokharna
*@version 1.0
*/
public class House {
    private static String string;

    /**
     * @param args String input
     */
    public static void main(String[] args) {
        Cat c1 = new Cat("Garfield");
        Cat c2 = new Cat("Tom", 10, 15);
        Cat c3 = new Cat("Meowth", 4.6, 5);
        Cat c4 = new Cat("Vidit", 18, 77);

        Mouse m1 = new Mouse(100, 3);
        Mouse m2 = new Mouse(60);
        Mouse m3 = new Mouse();

        c1.setBreed("1331 Cats");

        System.out.println(c1.toString());
        System.out.println(m2.toString());

        c3.increaseAge(8);
        c3.eat(m2);

        System.out.println(m2.toString());
        System.out.println(c2.toString());
        System.out.println(c4.toString());

        c3.increaseAge(4);

        string = c3.toString();
        System.out.println(string);
        System.out.println(c1.toString());
    }
}