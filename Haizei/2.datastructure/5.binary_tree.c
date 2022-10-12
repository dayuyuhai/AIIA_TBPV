#include<stdio.h>
#include<stdlib.h>
#include<time.h>

typedef struct Node {
    int val;
    struct Node *lchild, *rchild;
} Node;

typedef struct Tree {
    Node *root;
    int n;
} Tree;

Node *getNewNode(int val) {
    Node *p = (Node *)malloc(sizeof(Node));
    p->val = val;
    p->lchild = p->rchild = NULL;
    return p;
}

Tree *getNewTree() {
    Tree *t = (Tree *)malloc(sizeof(Tree));
    t->n = 0;
    t->root = NULL;
    return t;
}

Node *insertNode(Node *root, int val, int *ret) {
    if (root == NULL) {
        *ret = 1;
        return getNewNode(val);
    }
    if (root->val == val) return root;
    if (root->val > val) root->lchild = insertNode(root->lchild, val, ret);
    else root->rchild = insertNode(root->rchild, val, ret);
    return root;
}


void insert(Tree *tree, int val) {
    int flag = 0;
    tree->root = insertNode(tree->root, val, &flag);
    tree->n += flag;
    return ;
}


void clearNode(Node *node) {
    if (node == NULL) return ;
    clearNode(node->lchild);
    clearNode(node->rchild);
    free(node);
    return ;
}

void clearTree(Tree *tree) {
    clearNode(tree->root);
    free(tree);
    return ;
}

void outputNode(Node *root) {
    if (root == NULL) return ;
    printf("%d", root->val);
    if (root->lchild == NULL && root->rchild == NULL) return ;
    printf("(");
    outputNode(root->lchild);
    printf(",");
    outputNode(root->rchild);
    printf(")");
    return ;
}

void outputTree(Tree *tree) {
    printf("tree(%d) = ", tree->n);
    outputNode(tree->root);
    return ;
}

void preorderNode(Node *node) {
    if (node == NULL) return ;
    printf(" %d", node->val);
    preorderNode(node->lchild);
    preorderNode(node->rchild);
    return ;
}

void preorder(Tree *tree) {
    printf("preorder : ");
    preorderNode(tree->root);
    printf("\n");
    return ;
}

void inorderNode(Node *node) {
    if (node == NULL) return ;
    inorderNode(node->lchild);
    printf(" %d", node->val);
    inorderNode(node->rchild);
    return ;
}

void inorder(Tree *tree) {
    printf("inorder : ");
    inorderNode(tree->root);
    printf("\n");
    return ;
}

void postorderNode(Node *node) {
    if (node == NULL) return ;
    postorderNode(node->lchild);
    postorderNode(node->rchild);
    printf(" %d", node->val);
    return ;
}

void postorder(Tree *tree) {
    printf("postorder : ");
    postorderNode(tree->root);
    printf("\n");
    return ;
}

int main() {
	srand(time(0));
	#define MAX_OP 20
    Tree *tree = getNewTree();
	for (int i = 0; i < MAX_OP; i++) {
		int val = rand() % 100;
        insert(tree, val);
		outputTree(tree);
        printf("\n"); 
	}
    preorder(tree);
    inorder(tree);
    postorder(tree);
    clearTree(tree);

    return 0;
}