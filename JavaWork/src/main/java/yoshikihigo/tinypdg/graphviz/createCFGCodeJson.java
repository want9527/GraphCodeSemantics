package yoshikihigo.tinypdg.graphviz;

import com.ansj.vec.Word2vec;

import java.io.File;
import java.io.IOException;

public class createCFGCodeJson {
    //初始化Word2vec模型
    public static Word2vec vec = new Word2vec();
    public static void main(String[] args) throws IOException {
        vec.loadJavaModel("cfg_model.bin");
        //定义路径和图类型
        //String sourcePath = "./bigclonebenchdata";
        String sourcePath = "./test";

        String graphType = "cfg";
        WriterCfgToJsonTest w = new WriterCfgToJsonTest();
        //定义输出文件和路径
        File file_corpus_cfg= new File("./outPut_"+graphType+"/corpus/cfg_corpus.txt");

        File cfgDotFile = new File("./outPut_"+graphType+"/cfgDot");
        String codePath = "";
        String cfgPath = "";
        String[] cmd = {"-d", "./Test001.java", "-c", "test001cfg.dot"};
        //检查并删除现有的corpus文件
        if(!file_corpus_cfg.exists()){
            System.out.println("file_corpus_cfg文件不存在");
        }else{
            System.out.println("file_corpus_cfg存在文件");
            file_corpus_cfg.delete();
        }

        //扫描源代码文件
        Scan S = new Scan();
        S.Scanner(sourcePath,"java");
        //创建CFG dot文件目录c
        if(!cfgDotFile.exists()){//如果文件夹不存在
            cfgDotFile.mkdir();//创建文件夹
        }
        System.out.println("S.list.size()"+S.list.size());
        //处理每个源代码文件
        double start = System.currentTimeMillis();
        //遍历文件列表，为每个源代码文件生成对应的CFG dot文件路径并更新命令参数。
        for(int i=0; i<S.list.size(); i++){
            codePath = S.list.get(i);
            String[] pathParts = codePath.split(File.separator.equals("\\") ? "\\\\" : "/");
            StringBuilder javaCodePath = new StringBuilder(cfgDotFile + File.separator);
            for(int j = 4; j < pathParts.length - 1 ;j++){
                javaCodePath.append(pathParts[j]).append(File.separator);
            }

            File javaCodeFile = new File(javaCodePath.toString());
            if(!javaCodeFile.exists()){//如果文件夹不存在
                javaCodeFile.mkdirs();//创建文件夹
            }
            cfgPath = javaCodePath + pathParts[pathParts.length - 1] + ".cfg.dot";

            cmd[1] = codePath;
            cmd[3] = cfgPath;

            try{
                w.main(cmd);
                //System.out.println(i);
                //System.out.println(S.list.get(i));
            }catch (Exception e){
                System.out.println("触发异常 : "+e);
            }

        }
        double end = System.currentTimeMillis();
        System.out.println((end-start)/1000);
    }
}
