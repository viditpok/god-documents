/**
*@author Vidit Pokharna
*@version 1.0
*/
public class Narwhal extends Pet {
    /**
     *
     */
    private int hornLength;

    /**
     * @param name string
     * @param age int
     * @param painLevel int
     * @param hornLength int
     */
    public Narwhal(String name, int age, int painLevel, int hornLength) {
        super(name, age, painLevel);
        this.hornLength = hornLength;
    }

    /**
     *
     */
    public Narwhal() {
        this("Jelly", 19, 2, 7);
    }

    /**
     * @param p Pet
     */
    @Override
    public void playWith(Pet p) {
        int pain = super.getPainLevel();
        if (p instanceof Narwhal) {
            System.out.println("Who needs dogs and cats when we have each other");
            if (pain < 3) {
                super.setPainLevel(1);
            } else {
                super.setPainLevel(pain - 2);
            }
        } else {
            System.out.println("I live in the ocean so I can’t play with you");
            if (pain > 9) {
                super.setPainLevel(10);
            } else {
                super.setPainLevel(pain + 1);
            }
        }
    }

    /**
     * @return String
     */
    public String toString() {
        return super.toString() + " I have a horn that is " + hornLength + " feet long.";
    }

    /**
     * @param Object obj
     * @return boolean
     */
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Narwhal)) {
            return false;
        }
        Narwhal other = (Narwhal) obj;
        return (name.equals(other.name) && age == other.age
            && painLevel == other.painLevel && hornLength == other.hornLength);
    }
}