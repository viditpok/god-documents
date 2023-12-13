// I worked on the assignment alone, using only course-provided materials
/**
*@author Vidit Pokharna
*@version 1.0
*/
public abstract class Bender {
    /**
     *
     */
    protected String name;
    /**
     *
     */
    protected int strength;
    /**
     *
     */
    protected int health;

    /**
     * @param name string
     * @param strength int
     * @param health int
     */
    public Bender(String name, int strength, int health) {
        this.name = name;
        this.strength = strength;
        this.health = health;
    }

    /**
     * @return boolean
     */
    public boolean isAlive() {
        return (health > 0);
    }

    /**
     * @param b bender
     */
    public abstract void attack(Bender b);

    /**
     * @return String
     */
    public String toString() {
        String str = ("My name is " + this.name + ". I am a bender.");
        str += (" My strength is " + this.strength + " and my current ");
        str += ("health is " + this.health + ".");
        return str;
    }

    /**
     * @return name
     */
    public String getName() {
        return name;
    }

    /**
     * @return strength
     */
    public int getStrength() {
        return strength;
    }

    /**
     * @return health
     */
    public int getHealth() {
        return health;
    }

    /**
     * @param name setter
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * @param strength setter
     */
    public void setStrength(int strength) {
        this.strength = strength;
    }

    /**
     * @param health setter
     */
    public void setHealth(int health) {
        this.health = health;
    }

    /**
     * @param obj object
     * @return boolean
     */
    public boolean equals(Object obj) {
        if (!(obj instanceof Bender)) {
            return false;
        }
        Bender other = (Bender) obj;
        return (name.equals(other.name) && strength == other.strength && health == other.health);
    }
}