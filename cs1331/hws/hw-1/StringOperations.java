public class StringOperations {
    public static void main(String[] args) {

        String name = "Vidit";
        System.out.println(name);

        name = "A" + name.substring(1, name.length()-1) + "Z";
        System.out.println(name);

        String website = "www.gatech.edu";
        System.out.println(website);

        website = website.substring(4, website.length()-4) + "1331";
        System.out.println(website);
        
    }

}
