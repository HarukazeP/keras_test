package tamaki5_wiki;

import java.io.*;

public class WikiSentCheck {

	public static void main(String args[]) {
		long start = 0;
		long stop = 0;

		start = System.currentTimeMillis();
		//    	File file1 = new File("C:\\ruby_bundle\\wp2txt\\wp2txt-master\\MyOutput\\Enwiki.txt");
		//    	File file2 = new File("C:\\ruby_bundle\\wp2txt\\wp2txt-master\\MyOutput\\enwiki-latest-pages-articles.xml-0001.txt");
		//    	File file3 = new File("C:\\ruby_bundle\\WikiSentWithWords1.txt");
		//FilePro4.fileReadandDoForWiki(file);
		//FilePro4.fileReadandDoForWiki2(file);
		//FilePro4.CountSentFromFile3(file);

		//    	File file1 = new File("C:\\ruby_bundle\\Enwiki.txt");
		//    	FilePro5.ExcludeSentWithMark(file1);
		//    	System.out.println("1終わり");
		//
		//    	File file2 = new File("C:\\ruby_bundle\\WikiSentWithoutMark1.txt");
		//    	FilePro5.CheckIncludingWords(file2);
		//    	System.out.println("2終わり");
		//
		//    	File file3 = new File("C:\\ruby_bundle\\WikiSentWithWords1.txt");
		//    	FilePro5.CheckIncludingEndMark(file3);
		//    	System.out.println("2終わり");

		File file1 = new File("C:\\cygwin64\\home\\cspc13user\\WikiSentWithEndMark1.txt");
		FilePro5.makeMiniWiki(file1);

		stop = System.currentTimeMillis();
		long diff = (stop - start) / 1000;
		System.out.println("実行時間 : " + diff + "秒");

		//実行前にfileReadandDoForWiki2の中確認
		//if文の中，tureとかfalseとか
	}

}
