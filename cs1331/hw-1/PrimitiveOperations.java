//I worked on the assignment alone, using only course-provided materials.
public class PrimitiveOperations {
    
    public static void main(String[] args) {
        
        int one = 3;
        double two = 4.5;
        System.out.println(one);
        System.out.println(two);

        double three = two * one;
        System.out.println(three);

        double four = (double) one;
        System.out.println(four);

        int five = (int) two;
        System.out.println(five);

        char six = 'W';
        System.out.println(six);

        int seven = (int) six;
        six = (char) ((int) seven + 32);
        System.out.print(six);


    }

}
