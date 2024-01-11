/**
*@author Vidit Pokharna
*@version 1.0
*/
public abstract class Pet {
    /**
     *
     */
    protected String name;
    /**
     *
     */
    protected int age;
    /**
     *
     */
    protected int painLevel;

    /**
     * @param name String
     * @param age int
     * @param painLevel int
     */
    public Pet(String name, int age, int painLevel) {
        this.name = name;
        if (age < 1) {
            this.age = 1;
        } else if (age > 100) {
            this.age = 100;
        } else {
            this.age = age;
        }
        if (painLevel < 1) {
            this.painLevel = 1;
        } else if (painLevel > 10) {
            this.painLevel = 10;
        } else {
            this.painLevel = painLevel;
        }
    }

    /**
     * @param p Pet
     */
    public abstract void playWith(Pet p);

    /**
     * @return String
     */
    public String toString() {
        String ret = "My name is " + name + " and I am " + age + ". On a scale of one ";
        ret += "to ten my pain level is " + painLevel + ".";
        return ret;
    }

    /**
     * @param obj Object
     * @return boolean
     */
    public boolean equals(Object obj) {
        if (!(obj instanceof Pet)) {
            return false;
        }
        Pet other = (Pet) obj;
        return (name.equals(other.name) && age == other.age && painLevel == other.painLevel);
    }

    /**
     * @return String
     */
    public String getName() {
        return name;
    }

    /**
     * @param name String
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * @return int
     */
    public int getAge() {
        return age;
    }

    /**
     * @param age int
     */
    public void setAge(int age) {
        this.age = age;
        if (age < 1) {
            age = 1;
        } else if (age > 100) {
            age = 100;
        }
    }

    /**
     * @return int
     */
    public int getPainLevel() {
        return painLevel;
    }

    /**
     * @param painLevel int
     */
    public void setPainLevel(int painLevel) {
        this.painLevel = painLevel;
        if (painLevel < 1) {
            age = 1;
        } else if (painLevel > 10) {
            age = 10;
        }
    }
}