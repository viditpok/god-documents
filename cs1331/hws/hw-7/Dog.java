/**
*@author Vidit Pokharna
*@version 1.0
*/
public class Dog extends Pet implements Treatable {
    /**
     *
     */
    private String breed;

    /**
     * @param name string
     * @param age int
     * @param painLevel int
     * @param breed string
     */
    public Dog(String name, int age, int painLevel, String breed) {
        super(name, age, painLevel);
        this.breed = breed;
    }

    /**
     * @param breed string
     */
    public Dog(String breed) {
        this("Buzz", 6, 3, breed);
    }

    /**
     * @param Pet p
     */
    @Override
    public void playWith(Pet p) {
        int pain = super.getPainLevel();
        if (p instanceof Dog) {
            if (super.getPainLevel() < 4) {
                super.setPainLevel(1);
            } else {
                super.setPainLevel(pain - 3);
            }
            String s = "Woof! I love playing with other dogs so much that my pain level went ";
            s += "from " + pain + " to " + super.getPainLevel();
            System.out.println(s);
        } else if (p instanceof Cat) {
            Cat c = (Cat) p;
            if (c.hasStripes) {
                if (super.getPainLevel() > 8) {
                    super.setPainLevel(10);
                } else {
                    super.setPainLevel(super.getPainLevel() + 2);
                }
                System.out.println("AHHH! I thought you were a tiger!");
            } else {
                if (super.getPainLevel() > 1) {
                    super.setPainLevel(super.getPainLevel() - 1);
                }
                String s = "Woof. Cats without stripes are okay since they made my pain level ";
                s += "go from " + pain + " to " + super.getPainLevel();
                System.out.println(s);
            }
        }
    }

    /**
     *
     */
    @Override
    public void treat() {
        if (super.getPainLevel() > 3) {
            super.setPainLevel(super.getPainLevel() - 3);
        }
    }

    /**
     *
     */
    public void bark() {
        System.out.println("bark bark");
    }

    /**
     * @return string
     */
    public String toString() {
        String s = super.toString();
        String s1 = s.substring(0, s.indexOf('.'));
        String s2 = s.substring(s.indexOf('.'));
        String s3 = s1.substring(0, s1.indexOf(" and"));
        String s4 = s1.substring(s1.indexOf("I"));
        return (s3 + ", " + s4 + ", and I am a " + breed + s2 + " My age in human years is "
                + Treatable.convertDogToHumanYears(super.getAge()));
    }

    /**
     * @param Object obj
     * @return boolean
     */
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Dog)) {
            return false;
        }
        Dog other = (Dog) obj;
        return (name.equals(other.getName()) && age == other.getAge()
            && painLevel == other.getPainLevel() && breed.equals(other.getBreed()));
    }

    /**
     * @return string
     */
    public String getBreed() {
        return breed;
    }

    /**
     * @param breed string
     */
    public void setBreed(String breed) {
        this.breed = breed;
    }
}