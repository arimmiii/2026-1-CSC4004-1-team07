import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.indigo, useMaterial3: true),
      home: const LoginScreen(),
    );
  }
}

// --- 회원가입 ---
class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});
  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}
class _SignUpScreenState extends State<SignUpScreen> {
  final TextEditingController _id = TextEditingController();
  final TextEditingController _pw = TextEditingController();
  bool _isIdChecked = false;

  Future<void> _checkId() async {
    if (_id.text.isEmpty) return;
    try {
      final res = await http.get(Uri.parse('http://10.42.29.181:8000/check_id/${_id.text}'));
      final data = jsonDecode(res.body);
      if (data['available'] == true) {
        setState(() => _isIdChecked = true);
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("사용 가능한 아이디입니다."), backgroundColor: Colors.green));
      } else {
        setState(() => _isIdChecked = false);
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(data['message']), backgroundColor: Colors.orange));
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("서버 연결 실패")));
    }
  }

  Future<void> _reg() async {
    if (!_isIdChecked) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("아이디 중복 확인을 해주세요."), backgroundColor: Colors.redAccent));
      return;
    }
    try {
      await http.post(Uri.parse('http://10.42.29.181:8000/register'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"id": _id.text, "password": _pw.text}));
      if (!mounted) return;
      Navigator.pop(context);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("서버 연결 실패")));
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text("회원가입")),
    body: SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(children: [
        Row(children: [
          Expanded(child: TextField(controller: _id, decoration: const InputDecoration(labelText: "아이디"), onChanged: (v) => setState(() => _isIdChecked = false))),
          const SizedBox(width: 10),
          ElevatedButton(onPressed: _checkId, child: const Text("중복 확인")),
        ]),
        TextField(controller: _pw, decoration: const InputDecoration(labelText: "비번"), obscureText: true),
        const SizedBox(height: 30),
        SizedBox(width: double.infinity, child: ElevatedButton(onPressed: _reg, child: const Text("가입 완료"))),
      ]),
    ),
  );
}

// --- 로그인 ---
class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});
  @override
  State<LoginScreen> createState() => _LoginScreenState();
}
class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _id = TextEditingController();
  final TextEditingController _pw = TextEditingController();
  bool _doAutoLogin = false;

  @override
  void initState() {
    super.initState();
    _checkSavedLogin();
  }

  void _checkSavedLogin() async {
    final prefs = await SharedPreferences.getInstance();
    if (prefs.getBool('isLoggedIn') ?? false) {
      if (!mounted) return;
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (c) => const HomeScreen(isGuest: false)));
    }
  }

  void _login(bool guest) async {
    if (guest) {
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (c) => const HomeScreen(isGuest: true)));
      return;
    }
    try {
      final res = await http.post(Uri.parse('http://10.42.29.181:8000/login'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"id": _id.text, "password": _pw.text}));
      
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body);
        final prefs = await SharedPreferences.getInstance();
        await prefs.setInt('user_idx', data['user_idx']);
        await prefs.setBool('isLoggedIn', _doAutoLogin);

        if (!mounted) return;
        Navigator.pushReplacement(context, MaterialPageRoute(builder: (c) => const HomeScreen(isGuest: false)));
      } else {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("아이디 또는 비밀번호가 틀렸습니다."), backgroundColor: Colors.redAccent));
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("서버 연결 실패")));
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    body: Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(30),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.verified_user, size: 100, color: Colors.indigo),
            const Text("AI 뉴스 검증기", style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold)),
            const SizedBox(height: 20),
            TextField(controller: _id, decoration: const InputDecoration(labelText: "아이디")),
            TextField(controller: _pw, decoration: const InputDecoration(labelText: "비번"), obscureText: true),
            Row(children: [
              Checkbox(value: _doAutoLogin, onChanged: (v) => setState(() => _doAutoLogin = v!)),
              const Text("자동 로그인"),
            ]),
            const SizedBox(height: 20),
            SizedBox(width: double.infinity, child: ElevatedButton(onPressed: () => _login(false), child: const Text("로그인"))),
            TextButton(onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (c) => const SignUpScreen())), child: const Text("회원가입")),
            TextButton(onPressed: () => _login(true), child: const Text("로그인 없이 시작하기"))
          ],
        ),
      ),
    ),
  );
}

// --- 메인 (목록 필터링 적용) ---
class HomeScreen extends StatefulWidget {
  final bool isGuest;
  const HomeScreen({super.key, required this.isGuest});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}
class _HomeScreenState extends State<HomeScreen> {
  List<dynamic> news = [];
  String cat = '전체';
  final TextEditingController _s = TextEditingController();

  Future<void> load({String? c, String? q}) async {
    String url = 'http://10.42.29.181:8000/news?';
    if (c != null && c != '전체') url += 'category=$c&';
    if (q != null) url += 'search=$q';
    
    final res = await http.get(Uri.parse(url));
    if (res.statusCode == 200) {
      setState(() {
        // 백엔드에서 이미 처리하지만, 프론트에서도 빈 데이터가 오면 목록에서 제외
        news = jsonDecode(utf8.decode(res.bodyBytes));
      });
    }
  }
  @override
  void initState() { super.initState(); load(); }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: TextField(controller: _s, decoration: const InputDecoration(hintText: "뉴스 검색..."), onSubmitted: (v) => load(q: v))),
    drawer: Drawer(child: ListView(children: [
      DrawerHeader(decoration: const BoxDecoration(color: Colors.indigo), child: Text(widget.isGuest ? "게스트 모드" : "회원님 환영합니다", style: const TextStyle(color: Colors.white, fontSize: 18))),
      if (!widget.isGuest) 
        ListTile(leading: const Icon(Icons.favorite), title: const Text("마이페이지"), onTap: () => Navigator.push(context, MaterialPageRoute(builder: (c) => const MyPageScreen()))),
      if (widget.isGuest)
        ListTile(leading: const Icon(Icons.login), title: const Text("로그인 하러 가기"), onTap: () => Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (c) => const LoginScreen()), (r) => false)),
      if (!widget.isGuest) ...[
        const Divider(),
        ListTile(
          leading: const Icon(Icons.logout, color: Colors.red), 
          title: const Text("로그아웃", style: TextStyle(color: Colors.red)), 
          onTap: () async {
            final prefs = await SharedPreferences.getInstance();
            await prefs.clear();
            if (!mounted) return;
            Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (c) => const LoginScreen()), (r) => false);
          }
        ),
      ]
    ])),
    body: Column(children: [
      SingleChildScrollView(scrollDirection: Axis.horizontal, child: Row(children: ['전체', '정치', '경제', '사회', '생활/문화', 'IT/과학', '엔터', '스포츠'].map((category) => Padding(padding: const EdgeInsets.symmetric(horizontal: 4), child: ChoiceChip(label: Text(category), selected: cat == category, onSelected: (s) { setState(() => cat = category); load(c: category); }))).toList())),
      Expanded(
        child: news.isEmpty 
          ? const Center(child: Text("표시할 뉴스가 없습니다.")) // ✅ 빈 목록일 때 메시지
          : ListView.builder(itemCount: news.length, itemBuilder: (c, i) => ListTile(
              title: Text(news[i]['title']),
              subtitle: Row(children: [
                if (news[i]['category'] == '정치') Text("⚖️ 편향: ${news[i]['bias_score']}% ", style: const TextStyle(color: Colors.red)),
                if (news[i]['clickbait_score'] != null && news[i]['clickbait_score'] > 70) const Text("⚠️ 낚시주의! ", style: TextStyle(color: Colors.orange)),
                Text(news[i]['category'])
              ]),
              onTap: () => Navigator.push(context, MaterialPageRoute(builder: (c) => DetailScreen(idx: news[i]['idx'], isGuest: widget.isGuest))),
            ))
      )
    ]),
  );
}

// --- 마이페이지 ---
class MyPageScreen extends StatefulWidget {
  const MyPageScreen({super.key});
  @override
  State<MyPageScreen> createState() => _MyPageScreenState();
}
class _MyPageScreenState extends State<MyPageScreen> {
  List<dynamic> likes = [];
  void load() async {
    final prefs = await SharedPreferences.getInstance();
    final res = await http.get(Uri.parse('http://10.42.29.181:8000/user/${prefs.getInt('user_idx')}/likes'));
    if (res.statusCode == 200) setState(() { likes = jsonDecode(utf8.decode(res.bodyBytes)); });
  }
  @override
  void initState() { super.initState(); load(); }
  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text("내가 좋아요 한 뉴스")),
    body: ListView.builder(
      itemCount: likes.length, 
      itemBuilder: (c, i) => ListTile(
        title: Text(likes[i]['title']), 
        subtitle: Text(likes[i]['category']),
        onTap: () => Navigator.push(context, MaterialPageRoute(builder: (c) => DetailScreen(idx: likes[i]['idx'], isGuest: false))),
      )
    ),
  );
}

// --- 상세 페이지 ---
class DetailScreen extends StatefulWidget {
  final int idx;
  final bool isGuest;
  const DetailScreen({super.key, required this.idx, required this.isGuest});
  @override
  State<DetailScreen> createState() => _DetailScreenState();
}
class _DetailScreenState extends State<DetailScreen> {
  Map<String, dynamic>? data;
  void load() async {
    final res = await http.get(Uri.parse('http://10.42.29.181:8000/news/${widget.idx}'));
    if (res.statusCode == 200) setState(() { data = jsonDecode(utf8.decode(res.bodyBytes)); });
  }
  void like() async {
    final prefs = await SharedPreferences.getInstance();
    await http.post(Uri.parse('http://10.42.29.181:8000/like'), 
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"user_idx": prefs.getInt('user_idx'), "article_idx": widget.idx}));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("좋아요 목록에 추가되었습니다!")));
  }
  @override
  void initState() { super.initState(); load(); }
  @override
  Widget build(BuildContext context) {
    if (data == null) return const Scaffold(body: Center(child: CircularProgressIndicator()));
    return Scaffold(
      appBar: AppBar(title: const Text("기사 상세")),
      body: SingleChildScrollView(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(data!['title'], style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
        const Divider(),
        // ✅ 본문이 없을 경우 안내 메시지 표시 (혹시 모를 예외 처리)
        Text(data!['content'] != null && data!['content'].toString().isNotEmpty 
          ? data!['content'] 
          : "본문 내용을 불러올 수 없는 기사입니다."),
        const SizedBox(height: 20),
        if (data!['fact_check_results'] != null) ...[
          const Text("🔍 AI 팩트체크 결과", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.blue)),
          ...(data!['fact_check_results'] as List).map((f) => Card(child: ListTile(title: Text(f['target_text']), subtitle: Text(f['evidence']))))
        ]
      ])),
      floatingActionButton: widget.isGuest ? null : FloatingActionButton(onPressed: like, child: const Icon(Icons.favorite)),
    );
  }
}