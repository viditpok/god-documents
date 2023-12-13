/**
*@author Vidit Pokharna
*@version 1.0
*/
public class Vet {

    /**
     * @param p Pet
     */
    public static void inspectPet(Pet p) {
        System.out.println(p.toString());
        if (p instanceof Dog) {
            Dog d = (Dog) p;
            d.bark();
        }
    }

    /**
     * @param p Pet
     */
    public static void treatPet(Pet p) {
        if (p instanceof Dog || p instanceof Cat) {
            System.out.println("Welcome to the vet " + p.name);
            if (p instanceof Dog) {
                Dog d = (Dog) p;
                System.out.println("Wow what a cute dog!");
                d.treat();
                giveDogTreat(d);
            } else {
                Cat c = (Cat) p;
                c.treat();
            }
        } else {
            System.out.println("Sorry, we cannot treat " + p.getName());
        }
    }

    /**
     * @param d Dog
     */
    public static void giveDogTreat(Dog d) {
        int pain = d.getPainLevel();
        if (pain < 3) {
            d.setPainLevel(1);
        } else {
            d.setPainLevel(pain - 2);
        }
    }
}