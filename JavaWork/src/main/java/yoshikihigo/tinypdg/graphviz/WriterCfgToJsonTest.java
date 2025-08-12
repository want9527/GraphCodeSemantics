package yoshikihigo.tinypdg.graphviz;
import java.io.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.eclipse.jdt.core.dom.CompilationUnit;

import yoshikihigo.tinypdg.ast.TinyPDGASTVisitor;
import yoshikihigo.tinypdg.cfg.CFG;
import yoshikihigo.tinypdg.cfg.edge.CFGEdge;
import yoshikihigo.tinypdg.cfg.node.CFGControlNode;
import yoshikihigo.tinypdg.cfg.node.CFGNode;
import yoshikihigo.tinypdg.cfg.node.CFGNodeFactory;
import yoshikihigo.tinypdg.pdg.PDG;
import yoshikihigo.tinypdg.pdg.edge.PDGControlDependenceEdge;
import yoshikihigo.tinypdg.pdg.edge.PDGDataDependenceEdge;
import yoshikihigo.tinypdg.pdg.edge.PDGEdge;
import yoshikihigo.tinypdg.pdg.edge.PDGExecutionDependenceEdge;
import yoshikihigo.tinypdg.pdg.node.PDGControlNode;
import yoshikihigo.tinypdg.pdg.node.PDGMethodEnterNode;
import yoshikihigo.tinypdg.pdg.node.PDGNode;
import yoshikihigo.tinypdg.pdg.node.PDGNodeFactory;
import yoshikihigo.tinypdg.pdg.node.PDGParameterNode;
import yoshikihigo.tinypdg.pe.MethodInfo;
import yoshikihigo.tinypdg.pe.ProgramElementInfo;

import org.json.JSONObject;
import com.ansj.vec.Word2vec;
import java.io.File;
import java.io.IOException;
import org.apache.commons.io.FileUtils;

public class WriterCfgToJsonTest {
    public static String jsonPath = "";
    public static void main(String[] args) {

        try {
            //参数定义
            final Options options = new Options();

            {
                final Option d = new Option("d", "directory", true,
                        "target directory");
                d.setArgName("directory");
                d.setArgs(1);
                d.setRequired(true);
                options.addOption(d);
            }

            {
                final Option c = new Option("c", "ControlFlowGraph", true,
                        "control flow graph");
                c.setArgName("file");
                c.setArgs(1);
                c.setRequired(false);
                options.addOption(c);
            }

            {
                final Option p = new Option("p", "ProgramDepencencyGraph",
                        true, "program dependency graph");
                p.setArgName("file");
                p.setArgs(1);
                p.setRequired(false);
                options.addOption(p);
            }
            //解析命令行参数
            final CommandLineParser parser = new PosixParser();
            final CommandLine cmd = parser.parse(options, args);

            final File target = new File(cmd.getOptionValue("d"));
            jsonPath = cmd.getOptionValue("d");
            if (!target.exists()) {
                System.err
                        .println("specified directory or file does not exist.");
                System.exit(0);
            }
            //获取Java文件
            final List<File> files = getFiles(target);
            final List<MethodInfo> methods = new ArrayList<MethodInfo>();
            //解析Java文件并生成AST
            for (final File file : files) {
                final CompilationUnit unit = TinyPDGASTVisitor.createAST(file);
                final List<MethodInfo> m = new ArrayList<MethodInfo>();
                final TinyPDGASTVisitor visitor = new TinyPDGASTVisitor(
                        file.getAbsolutePath(), unit, methods);
                unit.accept(visitor);
                methods.addAll(m);
            }
            //构建控制流图（CFG）并写入文件
            if (cmd.hasOption("c")) {
                //System.out.println("building and outputing CFGs ...");
                final BufferedWriter writer = new BufferedWriter(
                        new FileWriter(cmd.getOptionValue("c")));

                writer.write("digraph CFG {");
                writer.newLine();

                final CFGNodeFactory nodeFactory = new CFGNodeFactory();

                int createdGraphNumber = 0;
                writer.write(" {");
                //拼接多个方法生成的json，要在头加“[”，尾部判断加“,”或者“]”
                boolean startFlag;
                boolean endFlag;
                for(int i = 0;i < methods.size();i++){
                //for (final MethodInfo method : methods) {
                    MethodInfo method = methods.get(i);
                    final CFG cfg = new CFG(method, nodeFactory);
                    cfg.build();
                    cfg.removeSwitchCases();
                    cfg.removeJumpStatements();
                    //生成的控制流图写入文件
                    startFlag = i == 0;
                    endFlag = i == methods.size() - 1;
                    writeMethodCFG(method.getName(),cfg, createdGraphNumber++, writer,startFlag,endFlag);
                }

                writer.write("}");

                writer.close();
            }

            //System.out.println("successfully finished.");

        } catch (Exception e) {
            System.err.println("Exception"+e.getMessage());
            System.exit(0);
        }
    }

    static private void writeMethodCFG(final String methodName,final CFG cfg,
                                       final int createdGraphNumber, final BufferedWriter writer,
                                        boolean startFlag,boolean endFlag)
            throws IOException {
        //写入子图开始标记
        writer.write("subgraph cluster");
        writer.write(Integer.toString(createdGraphNumber));
        writer.write(" {");
        writer.newLine();

        writer.write("label = \"");
        writer.write(getMethodSignature((MethodInfo) cfg.core));

        writer.write("\";");
        writer.newLine();
        //节点分配标签
        final SortedMap<CFGNode<? extends ProgramElementInfo>, Integer> nodeLabels = new TreeMap<CFGNode<? extends ProgramElementInfo>, Integer>();
        //System.out.println("cfg.getAllNodes()"+(cfg.getAllNodes()).size());/////////////////////////////////////////////////////////////
        for (final CFGNode<?> node : cfg.getAllNodes()) {
            nodeLabels.put(node, nodeLabels.size());
        }

        //创建JSON对象
        //CFG转为json
        //{"E":{"jsonNodes":{"0":"System . out . println () ; "},"jsonEdges":{}}}
        JSONObject methodCodeGraphJson = new JSONObject();
        JSONObject codeGraphJson = new JSONObject();
        JSONObject jsonNodes = new JSONObject();
        JSONObject jsonEdges = new JSONObject();
        //CFG转为向量
        //{"E":{"jsonNodes":{"0":[1,1,1,1,1,1...]},"jsonEdges":{}}}
        JSONObject methodCodeGraphJsonVec = new JSONObject();
        JSONObject codeGraphJsonVec = new JSONObject();
        JSONObject jsonNodesVec = new JSONObject();
        JSONObject jsonEdgesVec = new JSONObject();

        BufferedWriter cfg_corpus = new BufferedWriter(new FileWriter("./outPut_cfg/corpus/cfg_corpus.txt",true));
        //遍历节点并写入信息
        for (final Map.Entry<CFGNode<? extends ProgramElementInfo>, Integer> entry : nodeLabels.entrySet()) {
            //初始化和获取节点标签
            final CFGNode<? extends ProgramElementInfo> node = entry.getKey();

            final Integer label = entry.getValue();
            //createdGraphNumber：AST中第几个method
            //写入节点的唯一标识符，这里由 createdGraphNumber 和节点标签 label 组成。
            writer.write(Integer.toString(createdGraphNumber));
            writer.write(".");
            writer.write(Integer.toString(label));
            writer.write(" [style = filled, label = \"");
            //DATASET-去除多余空格-添加括号分号等符号的版本
            //处理节点文本和格式化
            String nodeString = node.getText().replace("\"", "\\\"")
                    .replace("\\\\\"", "\\\\\\\"").replace(";"," ; ")
                    .replace("\n", " \\n ").replace("\r", " \\r ")
                    .replace("(", " ( ").replace(")", " ) ")
                    .replace("[", " [ ").replace("]", " ] ")
                    .replace(" (  ) ", " () ").replace(" [  ] ", " [] ")
                    .replace("[][]", " [][] ").replace("="," = ")
                    .replace(".", " . ").replace(":"," : ").replace(","," , ")
                    .replace("+"," + ").replace("-"," - ").replace("+  +","++")
                    .replace("-  -","--").replace("~"," ~ ").replace("!"," ! ")
                    .replace("*"," * ").replace("/"," / ").replace("%"," % ")
                    .replace("<"," < ").replace(">"," > ").replace(">  >",">>")
                    .replace(">  >  >",">>>")
                    .replace("<  <","<<").replace("<  =","<=")
                    .replace(">  =",">=").replace("instanceof"," instanceof ").replace("=  =","==")
                    .replace("!  =","!=").replace("&"," & ").replace("|"," | ")
                    .replace("^"," ^ ").replace("&  &","&&").replace("|  |","||")
                    .replace("? :"," ?:").replace("+  =","+=")
                    .replace("-  =","-=").replace("*  =","*=").replace("/  =","/=")
                    .replace("%  =","%=").replace("&  =","&=").replace("|  =","|=").replace("/  /","//")
                    .replace("^  =","^=").replace("<  <  =","<<=").replace(">  >   =",">>=")
                    .replace(">  >  >  =",">>>=").replaceAll("\\s+"," ");


            writer.write(nodeString);
            writer.write("\"");

            // 写入文件
            cfg_corpus.write(nodeString);
            String[] code = nodeString.split("\\s+");
            //初始化词向量矩阵
            float[][] vectors = new float[code.length][createCFGCodeJson.vec.getWordVector("int").length];
            //计算词向量
            for(int i=0;i< code.length;i++){
                try{
                    vectors[i] = createCFGCodeJson.vec.getWordVector(code[i]);
                    if(vectors[i] == null){
                        vectors[i] = new float[32];
                    }
                    //System.out.println("vectors["+ i +"] = "+ Arrays.toString(vectors[i]));
                } catch (Exception e){
                    e.printStackTrace();
                    System.out.println("vectvectors[i]vectors[i]vectors[i]vectors[i]vectors[i]ors[i]");
                }
            }

            jsonNodes.put(Integer.toString(entry.getValue()),nodeString);
            jsonNodesVec.put(Integer.toString(entry.getValue()),vectors);
            //System.out.println("jsonNodes "+jsonNodes);


            final CFGNode<? extends ProgramElementInfo> enterNode = cfg.getEnterNode();
            final SortedSet<CFGNode<? extends ProgramElementInfo>> exitNodes = cfg.getExitNodes();

            if (enterNode == node) {
                writer.write(", fillcolor = aquamarine");
            } else if (exitNodes.contains(node)) {
                writer.write(", fillcolor = deeppink");
            } else {
                writer.write(", fillcolor = white");
            }

            if (node instanceof CFGControlNode) {
                writer.write(", shape = diamond");
            } else {
                writer.write(", shape = ellipse");
            }

            writer.write("];");
            writer.newLine();
        }

        cfg_corpus.newLine();
        cfg_corpus.close();

        //处理边并写入信息
        final SortedSet<CFGEdge> edges = new TreeSet<CFGEdge>();
        for (final CFGNode<?> node : cfg.getAllNodes()) {
            edges.addAll(node.getBackwardEdges());
            edges.addAll(node.getForwardEdges());
        }

        // create a json format for cfg

        //System.out.println("cfg.getAllEdges()"+edges.size());/////////////////////////////////////////////////////////////
        for (final CFGEdge edge : edges) {
            writer.write(Integer.toString(createdGraphNumber));
            writer.write(".");
            writer.write(Integer.toString(nodeLabels.get(edge.fromNode)));
            writer.write(" -> ");
            writer.write(Integer.toString(createdGraphNumber));
            writer.write(".");
            writer.write(Integer.toString(nodeLabels.get(edge.toNode)));
            writer.write(" [style = solid, label=\""
                    + edge.getDependenceString() + "\"];");

            // 写入文件
            jsonEdges.put(Integer.toString(nodeLabels.get(edge.fromNode))+"->"+Integer.toString(nodeLabels.get(edge.toNode)),edge.getDependenceString());


            String[] code = edge.getDependenceString().replace(";"," ;").replace("\n", "\\n").replace("\r", "\\r").split("\\s+");
            float[][] vectors = new float[code.length][createCFGCodeJson.vec.getWordVector("int").length];
            if(code[0].equals("")){
                Arrays.fill(vectors[0], 1);
            }else{
                for(int i=0;i< code.length;i++){
                    vectors[i] = createCFGCodeJson.vec.getWordVector(code[i]);
                    if(vectors[i] == null){
                        vectors[i] = new float[32];
                    }
                }
            }

            jsonEdgesVec.put(nodeLabels.get(edge.fromNode) +"->"+ nodeLabels.get(edge.toNode),vectors);

            writer.newLine();
        }


        codeGraphJson.put("jsonNodes",jsonNodes);
        codeGraphJson.put("jsonEdges",jsonEdges);
        methodCodeGraphJson.put(methodName,codeGraphJson);

        codeGraphJsonVec.put("jsonNodesVec",jsonNodesVec);
        codeGraphJsonVec.put("jsonEdgesVec",jsonEdgesVec);
        methodCodeGraphJsonVec.put(methodName,codeGraphJsonVec);
        //把json写入对应文件
        writeToFile(methodCodeGraphJson, methodCodeGraphJsonVec,startFlag,endFlag);

        writer.write("}");
        writer.newLine();
    }

    private static void writeToFile(JSONObject methodCodeGraphJson, JSONObject methodCodeGraphJsonVec,
                                    boolean startFlag,boolean endFlag) {
        try {
            //System.out.println("jsonPath CFG "+jsonPath);
            String[] pathParts = jsonPath.split(File.separator.equals("\\") ? "\\\\" : "/");
            StringBuilder codeJsonPath = new StringBuilder("./outPut_cfg/codeJson/");
            StringBuilder codeJsonVecPath = new StringBuilder("./outPut_cfg/codeJsonVec/");
            for(int j = 4; j < pathParts.length - 1 ;j++){
                codeJsonPath.append(pathParts[j]).append(File.separator);
                codeJsonVecPath.append(pathParts[j]).append(File.separator);
            }
            File codeJsonFile = new File(codeJsonPath.toString());
            if(!codeJsonFile.exists()){//如果文件夹不存在
                codeJsonFile.mkdirs();//创建文件夹
            }
            String start;
            String end;
            //通过判断，把一个java文件中的多个method的json拼接起来
            if(startFlag){
                start = "[";
            }else {
                start = "";
            }
            if(endFlag){
                end = "]";
            }else {
                end = ",";
            }
            FileUtils.write(new File(codeJsonPath+ File.separator + pathParts[pathParts.length - 1]+".json"),
                    start + methodCodeGraphJson.toString() + end, StandardCharsets.UTF_8, true);

            File codeJsonVecFile = new File(codeJsonVecPath.toString());
            if(!codeJsonVecFile.exists()){//如果文件夹不存在
                codeJsonVecFile.mkdirs();//创建文件夹
            }
            FileUtils.write(new File(codeJsonVecPath+ File.separator + pathParts[pathParts.length - 1]+".json"),
                    start + methodCodeGraphJsonVec.toString() + end, StandardCharsets.UTF_8, true);

            //FileUtils.write(new File("./outPut_cfg/codeJson/"+pathParts[3]+".json"), codeGraphJson.toString(), StandardCharsets.UTF_8, true);
            //FileUtils.write(new File("./outPut_cfg/codeJsonVec/"+pathParts[3]+".json"), codeGraphJsonVec.toString(), StandardCharsets.UTF_8, true);
        } catch (IOException e) {
            //System.out.println("ssss");
            e.printStackTrace();
        }
    }

    static private List<File> getFiles(final File file) {

        final List<File> files = new ArrayList<File>();

        if (file.isFile() && file.getName().endsWith(".java")) {
            files.add(file);
        }

        else if (file.isDirectory()) {
            for (final File child : file.listFiles()) {
                final List<File> children = getFiles(child);
                files.addAll(children);
            }
        }

        return files;
    }

    static private String getMethodSignature(final MethodInfo method) {

        final StringBuilder text = new StringBuilder();

        text.append(method.name);
        text.append(" <");
        text.append(method.startLine);
        text.append("...");
        text.append(method.endLine);
        text.append(">");

        return text.toString();
    }
}
